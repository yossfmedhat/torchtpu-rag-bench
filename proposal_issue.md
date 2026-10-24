<!-- File after Phase 1. Fill every [bracket] with measured numbers from
main.jsonl / buckets4.jsonl. Delete any claim the data does not support. -->

**Title:** Proposal: embedding + reranking (RAG retrieval) example

### What

A single-device inference example under `examples/rag_retrieval/` that:

1. Embeds a small mixed Arabic/English corpus with `BAAI/bge-m3` (dense CLS
   vectors) on TPU, eager and `torch.compile(backend="tpu")`.
2. Retrieves top-k passages for a query by cosine similarity.
3. Reranks them with the cross-encoder `BAAI/bge-reranker-v2-m3`.
4. Checks the TPU outputs against a CPU float32 reference
   (`torch.testing.assert_close` / cosine), following
   `docs/numerical_precision.md`.

Layout follows `examples/simple_pytorch/`: one script, a README with the run
command and expected output. Target size about 100–150 lines.

### Why

The examples cover training (FSDP, tensor parallel) and LLM decoding. None
covers encoder-only inference, which is what most retrieval pipelines run.
Encoder batches have a different shape problem from decoding: sequence length
varies per batch rather than growing by one token, so the KV-cache walkthrough
in `docs/dynamic_shapes.md` does not map onto it directly.

### What I measured (TPU [v5e-1], torch_tpu [0.2.0])

| Padding | compile-like calls over [12] batches | steady latency | end-to-end texts/s |
|---|---|---|---|
| pad to longest in batch | [ ] | [ ] | [ ] |
| pad to next multiple of 128 | [ ] | [ ] | [ ] |
| pad to 512 | [ ] | [ ] | [ ] |

Accuracy vs CPU fp32: min cosine [ ] at default precision, [ ] at `"highest"`;
reranker top-1 agreement [ ].

### Advice the example would show

[Write this from the data. The expected shape: pad to multiples of 128 (the
documented bucket size) and set `num_dynamic_buckets` to cover the length range,
or `mark_dynamic` on the seq dim with explicit min/max; sort-by-length batching
to cut padding waste.]

### Questions for maintainers

- Is `examples/` the right place, or would you prefer a docs notebook?
- OK to depend on `transformers` in an example, or should the encoder be
  defined inline?
- Any preference on model size for CI or doc tests (cf. #5535)? I can swap
  in a small multilingual encoder if downloading BGE-M3 (2.2 GB) is too heavy.

I'd like to implement this if it fits.
