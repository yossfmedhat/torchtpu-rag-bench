# Running Phase 0–1 on Colab

Runtime > Change runtime type > TPU (v5e-1 or v6e-1). Then:

```bash
!git clone https://github.com/yossfmedhat/torchtpu-rag-bench
%cd torchtpu-rag-bench
```

## Cell 1: install and record the environment

```bash
!python --version
!pip list 2>/dev/null | grep -iE "^(torch|jax|jaxlib|libtpu|transformers) "
!pip install torch_tpu==0.2.0 transformers --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -20
!pip list 2>/dev/null | grep -iE "^(torch|torch.tpu|jax|jaxlib|libtpu|transformers|torchvision|torchaudio) "
```

Restart the runtime after installing. Watch for:

- pip replacing Colab's preinstalled `torch` (torch_tpu 0.2.0 needs
  `torch>=2.11,<=2.15`) and leaving `torchvision`/`torchaudio` mismatched.
- `libtpu` conflicts with Colab's preinstalled `jax`.

## Cell 2: README quickstart (from `examples/README.md`)

```python
import torch
device = torch.accelerator.current_accelerator()
x = torch.ones(2, 2, device=device)
print(f"Tensor on TPU: {x}")
print(f"Result on TPU: {x + x}")
```

If this fails, that is issue #1. Copy: runtime type, `python --version`, the
`pip list` lines above, and the full traceback. Fallbacks: Kaggle TPU (see the
open issue #3047 about wheels on Kaggle first), then TPU Research Cloud.

## Cell 3: sanity run (≈ minutes)

```bash
!python rag_tpu_bench.py --tasks embed --batch-sizes 8 --modes eager --strategies max --num-batches 3 --out sanity.jsonl
```

## Cell 4: the main matrix

```bash
!python rag_tpu_bench.py --out main.jsonl
```

## Cell 5: sweeps that explain the main results

```bash
# Is low cosine just TPU's default bf16 matmul? (docs/numerical_precision.md)
!python rag_tpu_bench.py --matmul-precision highest --modes eager --batch-sizes 8 --out prec_highest.jsonl
# bf16 weights end to end
!python rag_tpu_bench.py --dtype bfloat16 --out bf16.jsonl
# More dynamic buckets for the "longest" strategy (docs/dynamic_shapes.md)
!python rag_tpu_bench.py --modes compile --compile-options '{"num_dynamic_buckets": 4}' --out buckets4.jsonl
# Static shapes: one compile per distinct width
!python rag_tpu_bench.py --modes compile --compile-dynamic false --out static.jsonl
```

Download all `*.jsonl` files. Each starts with an `env` line; error rows hold
the full traceback.

## Reading the results: bug or expected?

Check these before filing. They are documented behaviour, not bugs:

| Observation | Why | Check |
|---|---|---|
| fp32 `min_cos` a bit below 0.999 | Default fp32 matmul precision on TPU is `"medium"` (one bf16 pass) | Rerun with `--matmul-precision highest`. Still low → real issue |
| `longest` compiles 2–3 times then stabilises | Bounded dynamism: static compile, then dynamic bucket, then layout recompile | `docs/dynamic_shapes.md` "Example" section |
| Recompile once width passes 128 | Default `num_dynamic_buckets=1` | `buckets4.jsonl` should remove it |
| `max` slowest steady state at small inputs | Pays FLOPs for 512 tokens | Expected |
| Bucket executables recompiled after process restart | No persistent cache for bounded-dynamic executables | Documented limitation |

Likely real issues:

- `min_cos < 0.999` with `--matmul-precision highest`, or `top1_agree < 1.0`
  on the real reranker. Bisect by layer: GELU has an open fp32 report (#5700),
  so compare one encoder layer's output on CPU vs TPU first.
- Crashes or lowering errors in compile mode, especially with an attention mask
  and dynamic seq length. Compare against #5479 and #5674 before filing.
- `compile_like_calls` that keep growing within one bucket.
- Eager mode recompiling per shape (compare #5086).
