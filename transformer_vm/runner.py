#!/usr/bin/env python3
"""Run WASM programs through the transformer.

Builds the model weights automatically if no model.bin exists.
Uses the C++ inference engine by default; pass --python for Python fallback.

For graph evaluation (exact arithmetic, no weights), use wasm-eval instead.
"""

import argparse
import glob
import logging
import os
import platform
import subprocess
import time

logger = logging.getLogger(__name__)


# ── Python model inference ────────────────────────────────────────


def run_model_program(
    model,
    all_tokens,
    tok_to_idx_map,
    program_file,
    ref_file=None,
    max_new_tokens=2000,
    verbose=False,
    cache_class=None,
):
    """Run a program through a saved transformer model.

    The input file should contain the exact tokens the model expects:
    the full .txt for a universal model, or a _spec.txt for a specialized one.

    Returns (ok, n_tok, n_ops) where ok is True if output matches reference.
    """
    import torch

    with open(program_file) as f:
        tokens = f.read().split()
    idx_seq = [tok_to_idx_map[t] for t in tokens]

    kwargs = dict(max_new_tokens=max_new_tokens)
    if cache_class is not None:
        kwargs["cache_class"] = cache_class
    result = model.generate_with_cache(torch.tensor([idx_seq], dtype=torch.long), **kwargs)
    predicted = [all_tokens[i] for i in result[0].tolist()]
    n_tok = len(predicted)
    n_ops = sum(1 for t in predicted if "commit" in t or t == "branch_taken")

    if verbose:
        logger.info("  Tokens: %s", " ".join(predicted))

    if ref_file and os.path.exists(ref_file):
        with open(ref_file) as f:
            ref_tokens = f.read().split()
        for i in range(min(len(predicted), len(ref_tokens))):
            if predicted[i] != ref_tokens[i]:
                logger.warning(
                    "  MISMATCH at position %d: predicted=%s, expected=%s",
                    i,
                    predicted[i],
                    ref_tokens[i],
                )
                return False, n_tok, n_ops
        if len(predicted) > len(ref_tokens):
            logger.warning(
                "  MISMATCH at position %d: predicted=%s, expected=<END>",
                len(ref_tokens),
                predicted[len(ref_tokens)],
            )
            return False, n_tok, n_ops
        if len(predicted) < len(ref_tokens):
            logger.info("  output truncated: %d/%d tokens", n_tok, len(ref_tokens))
        return True, n_tok, n_ops

    return True, n_tok, n_ops


# ── C++ engine ────────────────────────────────────────────────────

_CPP_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "transformer.cpp")
_CPP_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transformer")


def _find_libomp_prefix():
    """Find libomp install prefix on macOS (homebrew)."""
    for prefix in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
        if os.path.isfile(os.path.join(prefix, "lib", "libomp.dylib")):
            return prefix
    return None


def _build_cpp_engine():
    """Build the C++ inference engine if not already built."""
    binary = os.path.abspath(_CPP_BINARY)
    source = os.path.abspath(_CPP_SOURCE)
    if os.path.exists(binary):
        return binary
    if not os.path.exists(source):
        return None

    logger.info("[engine] Compiling C++ inference engine...")
    attn_dir = os.path.join(os.path.dirname(source), "..", "attention")
    if platform.system() == "Darwin":
        omp_prefix = _find_libomp_prefix()
        cmd = [
            "clang++",
            "-std=c++17",
            "-O3",
            "-framework",
            "Accelerate",
            "-I",
            attn_dir,
        ]
        if omp_prefix:
            cmd += ["-Xclang", "-fopenmp",
                    f"-I{omp_prefix}/include", f"-L{omp_prefix}/lib", "-lomp"]
        cmd += [source, "-o", binary]
    else:
        cmd = ["g++", "-std=c++17", "-O3", "-fopenmp",
               "-I", attn_dir, source, "-o", binary,
               "-lopenblas"]
    try:
        subprocess.check_call(cmd)
        logger.info("[engine] Built: %s", binary)
        return binary
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("[engine] Could not build C++ engine: %s", e)
        return None


def run_cpp_engine(binary, model_path, files, brute=False):
    """Run programs through the C++ inference engine.

    Returns True if all programs with refs passed.
    """
    cmd = [binary, model_path]
    if brute:
        cmd.append("--brute")
    cmd += files
    result = subprocess.run(cmd)
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model.bin")


def _ensure_model(model_path):
    """Build model weights if they don't exist."""
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        logger.info("[model] Loading weights from %s", model_path)
        return model_path
    logger.info("[model] Weights not found at %s", model_path)
    logger.info("[model] Solving MILP schedule and constructing weights...")
    from transformer_vm.build import build
    from transformer_vm.model.weights import save_weights

    model, all_tokens, tok_to_idx_map = build()
    save_weights(model, all_tokens, model_path)
    logger.info("[model] Saved weights to %s", model_path)
    return model_path


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run WASM programs through the transformer.")
    parser.add_argument("files", nargs="*", help="Program .txt files to run")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to model weights (.bin); auto-built if missing",
    )
    parser.add_argument(
        "--python", action="store_true", help="Force Python inference (default uses C++ engine)"
    )
    parser.add_argument(
        "--nohull",
        action="store_true",
        help="Use brute-force O(n) attention (StandardKVCache) instead of hull cache",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print the full generated token sequence"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50000, help="Max tokens to generate per program"
    )
    args = parser.parse_args()

    # Step 1: Compile C examples to WASM token files if needed
    logger.info("[compile] Checking for compiled programs...")
    from transformer_vm.compilation.compile_wasm import ensure_data

    ensure_data()

    files = args.files
    if not files:
        from transformer_vm._paths import DATA_DIR

        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
        files = [f for f in files if not any(s in f for s in ("_ref", "_spec"))]
    logger.info("[compile] %d program(s) to run", len(files))

    # Step 2: Build transformer weights if needed
    model_path = _ensure_model(args.model)

    # Step 3: Build C++ inference engine and run
    if not args.python:
        binary = _build_cpp_engine()
        if binary is None:
            raise RuntimeError(
                "Could not build C++ inference engine. Use --python to run with Python instead."
            )
        logger.info("[engine] Running %d program(s) via C++ engine", len(files))
        ok = run_cpp_engine(binary, model_path, files, brute=args.nohull)
        if not ok:
            raise SystemExit(1)
        return

    # Python inference fallback
    logger.info("[engine] Running %d program(s) via Python inference", len(files))
    from transformer_vm.model.weights import load_weights

    model, all_tokens, tok_to_idx_map = load_weights(model_path)

    if args.nohull:
        from transformer_vm.attention import StandardKVCache

        cache_class = StandardKVCache
    else:
        from transformer_vm.attention import HullKVCache

        cache_class = HullKVCache

    passed = failed = skipped = 0
    total_tokens = total_ops = 0
    total_time = 0.0

    for prog_file in files:
        if "_ref" in prog_file:
            continue
        name = os.path.basename(prog_file).replace(".txt", "")
        ref_file = prog_file.replace(".txt", "_ref.txt")
        has_ref = os.path.exists(ref_file)

        t0 = time.time()
        ok, n_tok, n_ops = run_model_program(
            model,
            all_tokens,
            tok_to_idx_map,
            prog_file,
            ref_file if has_ref else None,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            cache_class=cache_class,
        )
        dt = time.time() - t0
        total_tokens += n_tok
        total_ops += n_ops
        total_time += dt

        if has_ref:
            status = "PASS" if ok else "FAIL"
            logger.info(
                "%s: %s  %d tok, %d ops in %.2fs (%.0f tok/s)",
                name,
                status,
                n_tok,
                n_ops,
                dt,
                n_tok / max(dt, 1e-9),
            )
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            logger.info(
                "%s: RAN   %d tok, %d ops in %.2fs (%.0f tok/s)",
                name,
                n_tok,
                n_ops,
                dt,
                n_tok / max(dt, 1e-9),
            )
            skipped += 1

    logger.info("%d passed, %d failed, %d no-ref", passed, failed, skipped)
    if total_time > 0:
        from transformer_vm.model.weights import flops_per_token

        fpt = flops_per_token(model)
        logger.info("Benchmark: %d tok, %d ops, %.2fs", total_tokens, total_ops, total_time)
        logger.info(
            "  %.0f tok/s, %.0f wasm-ops/s",
            total_tokens / total_time,
            total_ops / total_time if total_ops else 0,
        )
        logger.info("  %.1fM FLOPs/tok", fpt / 1e6)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
