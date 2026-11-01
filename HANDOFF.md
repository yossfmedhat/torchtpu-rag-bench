# Handoff: TorchTPU first-contribution plan

Goal: a visible first contribution to google-pytorch/torch_tpu, built on
multilingual (Arabic + English) embedding and reranking for RAG.

## Done

- Fork: github.com/yossfmedhat/torch_tpu (upstream: google-pytorch/torch_tpu).
- This repo: benchmark harness `rag_tpu_bench.py`, Colab steps `COLAB.md`,
  issue drafts `issue_bug_template.md` and `proposal_issue.md`.
- Upstream rules (CONTRIBUTING.md): small PRs, open an issue before any
  non-trivial PR, Google CLA required, AI must not be listed as co-author
  (use an `Assisted-by:` trailer instead). PR body uses
  `.github/pull_request_template.md` (Description, Test Plan).

## First Colab run (v5e-1, Python 3.13.16, torch 2.14.1+cpu, torch_tpu 0.2.0, libtpu 0.0.50)

- README quickstart works on Colab.
- BGE-M3 embeddings: min cosine vs CPU fp32 0.99995 (default precision).
- bge-reranker-v2-m3: top-1 agreement 1.00, max |logit diff| 0.163
  (to check with `--matmul-precision highest`).
- torch.compile with fixed 512 padding: 29 ms/batch vs 70 ms eager, 11 s
  first compile.
- **Bug:** torch.compile(backend="tpu", dynamic=None) fails once the seq dim
  becomes dynamic (pad-to-longest and pad-to-128 both fail), for both models:
  `layer_norm` lowering, `'stablehlo.broadcast_in_dim' op result #0 must be
  statically shaped or single bounded dimension tensor ... but got
  'tensor<8x?x1024xf32>'` at modeling_xlm_roberta.py:120 (embeddings
  LayerNorm). Error: `NotImplementedError: Failed to lower FX graph with
  dynamic shapes to MLIR`. Related upstream issues: #5479, #5674, #5678.
- Eager mode, pad-to-longest is 12x slower than pad-to-128 (729 vs 62 ms)
  because every new shape compiles. Advice for the example, not a bug.

## Next

1. On Colab, run (same runtime as the first run, or reinstall first):
   ```
   %cd /content/torchtpu-rag-bench
   !git pull -q
   !python repro_layernorm_dynamic.py 2>&1 | grep -E "^(OK|FAIL|torch)"
   !python rag_tpu_bench.py --modes compile --compile-dynamic false --batch-sizes 8 --num-batches 6 --out static.jsonl 2>&1 | grep -E "^(embed|rerank)"
   !python rag_tpu_bench.py --tasks rerank --modes eager --strategies bucket --batch-sizes 8 --num-batches 6 --matmul-precision highest --out prec.jsonl 2>&1 | grep -E "^rerank"
   ```
2. From the smallest FAIL case in `repro_layernorm_dynamic.py`, write the
   bug issue with `issue_bug_template.md`. Search upstream issues for
   "broadcast_in_dim" and "bounded" first.
3. After that, post `proposal_issue.md` with the measured numbers, and wait
   for a maintainer before writing the example PR.
