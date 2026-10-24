# torchtpu-rag-bench

Benchmarks multilingual (Arabic + English) embedding and reranking on
[TorchTPU](https://github.com/google-pytorch/torch_tpu), compared against a
CPU float32 reference.

- Models: `BAAI/bge-m3` (dense embeddings), `BAAI/bge-reranker-v2-m3`
  (cross-encoder)
- Modes: eager and `torch.compile(backend="tpu")`
- Padding strategies: pad to the longest text in the batch, to the next
  multiple of 128, or to a fixed 512, to see how varying sequence lengths
  affect compilation
- Metrics: cosine similarity / logit diff against CPU, first-call time,
  compile-like calls, steady-state latency, end-to-end throughput

Run instructions: [COLAB.md](COLAB.md).

```bash
pip install torch_tpu==0.2.0 transformers --extra-index-url https://download.pytorch.org/whl/cpu
python rag_tpu_bench.py --out results.jsonl
# CPU smoke test with a tiny random model, no downloads:
python rag_tpu_bench.py --tiny --device cpu --compile-backend aot_eager
```

`issue_bug_template.md` and `proposal_issue.md` are drafts for upstream issues.
