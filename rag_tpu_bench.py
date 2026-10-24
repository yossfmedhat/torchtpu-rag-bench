"""Phase 1 harness: BGE-M3 embeddings + bge-reranker-v2-m3 on TorchTPU vs CPU.

For every (task, mode, padding strategy, batch size) it measures:
  * correctness against a CPU float32 reference on identical inputs
      embed:  min / mean cosine similarity of the dense (CLS, L2-normed) vectors
      rerank: max |logit diff| and top-1 agreement per query group
  * first-call time, number of compile-like calls (latency > 10x steady median),
    steady-state median latency and throughput, total wall time.

Padding strategies show how sequence-length dynamism interacts with XLA:
  longest   pad each batch to its own longest sequence (shape changes often)
  bucket    pad to the next multiple of --bucket (default 128, the TPU lane width)
  max       always pad to --max-len (one shape, wasted FLOPs)

Every failure is caught and written to results with its full traceback: each
one is a candidate GitHub issue.

Examples:
  # real models on TPU (Colab / TPU VM)
  python rag_tpu_bench.py --out results_tpu.jsonl
  # CPU-only smoke test of the harness with a tiny random XLM-R (no download)
  python rag_tpu_bench.py --tiny --device cpu --compile-backend aot_eager
"""

import argparse
import copy
import json
import math
import platform
import random
import statistics
import time
import traceback

import torch
import torch.nn.functional as F

AR = [
    "ما هي عاصمة المملكة العربية السعودية؟",
    "الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها.",
    "كيف تعمل نماذج التضمين في البحث الدلالي؟",
    "تحوّل نماذج التضمين النصوص إلى متجهات بحيث تكون النصوص المتقاربة في"
    " المعنى قريبة من بعضها.",
    "يقع نهر النيل في شمال شرق أفريقيا ويعد من أطول الأنهار في العالم.",
    "تُستخدم وحدات معالجة الموترات لتسريع تدريب نماذج التعلم العميق.",
    "يعيد نموذج إعادة الترتيب تقييم المقاطع المسترجعة حسب صلتها بالسؤال.",
    "القاهرة مدينة كبيرة تقع على ضفاف نهر النيل.",
]
EN = [
    "What is the capital of Saudi Arabia?",
    "Riyadh is the capital and largest city of Saudi Arabia.",
    "How do embedding models work in semantic search?",
    "Embedding models map text to vectors so that texts with similar meaning"
    " end up close together.",
    "The Nile flows through north-eastern Africa and is one of the longest"
    " rivers in the world.",
    "Tensor Processing Units are used to accelerate deep learning training.",
    "A reranker rescores retrieved passages by their relevance to the query.",
    "Cairo is a large city on the banks of the Nile.",
]
# (query, relevant passage, distractors...) -- cross-lingual on purpose.
RERANK_GROUPS = [
    (AR[0], EN[1], EN[4], AR[5], EN[7]),
    (EN[2], AR[3], EN[1], AR[4], EN[5]),
    (AR[2], EN[3], AR[1], EN[4], AR[7]),
    (EN[0], AR[1], EN[3], AR[4], EN[6]),
    (AR[4], EN[4], AR[1], EN[2], AR[5]),
    (EN[6], AR[6], EN[0], AR[3], EN[5]),
]


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="tpu")
  p.add_argument("--tasks", default="embed,rerank")
  p.add_argument("--modes", default="eager,compile")
  p.add_argument("--strategies", default="longest,bucket,max")
  p.add_argument("--batch-sizes", default="1,8,32")
  p.add_argument("--max-len", type=int, default=512)
  p.add_argument("--bucket", type=int, default=128)
  p.add_argument("--num-batches", type=int, default=12)
  p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  p.add_argument("--matmul-precision", default="medium",
                 choices=["medium", "high", "highest"])
  p.add_argument("--compile-backend", default="tpu")
  p.add_argument("--compile-dynamic", default="none",
                 choices=["none", "false"])
  p.add_argument("--compile-options", default="{}",
                 help='JSON, e.g. {"num_dynamic_buckets": 4}')
  p.add_argument("--embed-model", default="BAAI/bge-m3")
  p.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
  p.add_argument("--tiny", action="store_true",
                 help="random tiny XLM-R + byte tokenizer, no downloads")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="results.jsonl")
  return p.parse_args()


def sync(device):
  if device.type != "cpu":
    torch.accelerator.synchronize()


def environment(args):
  env = {
      "python": platform.python_version(),
      "torch": torch.__version__,
      "device": args.device,
      "dtype": args.dtype,
      "matmul_precision": args.matmul_precision,
  }
  for mod in ("torch_tpu", "transformers", "libtpu"):
    try:
      env[mod] = __import__(mod).__version__
    except Exception as e:  # pylint: disable=broad-except
      env[mod] = f"unavailable ({type(e).__name__})"
  if args.device != "cpu":
    try:
      env["device_props"] = str(torch.tpu.get_device_properties(0))
    except Exception as e:  # pylint: disable=broad-except
      env["device_props"] = f"unavailable ({type(e).__name__}: {e})"
  return env


class ByteTokenizer:
  """Stand-in for the XLM-R tokenizer in --tiny mode (ids 0..2 reserved)."""

  def __call__(self, text, text_pair=None, max_length=512):
    ids = [0] + [b + 3 for b in text.encode()][: max_length - 2] + [2]
    if text_pair is not None:
      ids = ids[:-1] + [2, 2] + [b + 3 for b in text_pair.encode()] + [2]
      ids = ids[: max_length - 1] + [2]
    return ids


def load(args, task):
  from transformers import (AutoModel, AutoModelForSequenceClassification,
                            AutoTokenizer, XLMRobertaConfig,
                            XLMRobertaForSequenceClassification,
                            XLMRobertaModel)
  if args.tiny:
    torch.manual_seed(args.seed)
    cfg = XLMRobertaConfig(vocab_size=259, hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=128,
                           max_position_embeddings=args.max_len + 2,
                           pad_token_id=1, num_labels=1)
    cls = (XLMRobertaModel if task == "embed"
           else XLMRobertaForSequenceClassification)
    model, tok = cls(cfg), ByteTokenizer()
  else:
    name = args.embed_model if task == "embed" else args.rerank_model
    cls = AutoModel if task == "embed" else AutoModelForSequenceClassification
    model = cls.from_pretrained(name, torch_dtype=torch.float32)
    hf_tok = AutoTokenizer.from_pretrained(name)

    def tok(text, text_pair=None, max_length=args.max_len):
      return hf_tok(text, text_pair, truncation=True,
                    max_length=max_length)["input_ids"]
  return model.eval(), tok


def make_inputs(args, task, tok, batch_size):
  """Seeded stream of batches with a spread of real token lengths."""
  rng = random.Random(args.seed)
  pool = AR + EN
  batches = []
  for _ in range(args.num_batches):
    seqs, groups = [], []
    if task == "embed":
      for _ in range(batch_size):
        n = rng.choice([1, 1, 1, 2, 2, 3, 4, 8, 16])  # skewed toward short texts
        seqs.append(tok(" ".join(rng.choice(pool) for _ in range(n)),
                        max_length=args.max_len))
    else:
      while len(seqs) < batch_size:
        q, *passages = rng.choice(RERANK_GROUPS)
        reps = rng.choice([1, 1, 2, 4, 8])
        start = len(seqs)
        for p in passages[: batch_size - len(seqs)]:
          seqs.append(tok(q, " ".join([p] * reps), max_length=args.max_len))
        groups.append((start, len(seqs)))
    batches.append((seqs, groups))
  return batches


def pad(seqs, strategy, args):
  longest = max(len(s) for s in seqs)
  width = {"longest": longest,
           "bucket": min(args.max_len,
                         math.ceil(longest / args.bucket) * args.bucket),
           "max": args.max_len}[strategy]
  ids = torch.ones(len(seqs), width, dtype=torch.long)  # XLM-R pad id is 1
  mask = torch.zeros(len(seqs), width, dtype=torch.long)
  for i, s in enumerate(seqs):
    ids[i, : len(s)] = torch.tensor(s)
    mask[i, : len(s)] = 1
  return ids, mask


def forward_fn(task):
  def embed(model, ids, mask):
    cls = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    return F.normalize(cls.float(), dim=-1)

  def rerank(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask).logits[:, 0].float()

  return embed if task == "embed" else rerank


def compare(task, out, ref, groups):
  if task == "embed":
    cos = F.cosine_similarity(out, ref, dim=-1)
    return {"min_cos": cos.min().item(), "mean_cos": cos.mean().item()}
  top1 = [int(out[a:b].argmax() == ref[a:b].argmax()) for a, b in groups]
  return {"max_abs_logit_diff": (out - ref).abs().max().item(),
          "top1_agree": sum(top1) / max(len(top1), 1)}


def run_config(args, task, mode, strategy, batches, refs, model_cpu, device):
  fwd = forward_fn(task)
  dtype = getattr(torch, args.dtype)
  # Module.to() is in place; copy so the CPU float32 reference stays exact.
  model = copy.deepcopy(model_cpu).to(device=device, dtype=dtype)
  if mode == "compile":
    torch._dynamo.reset()  # pylint: disable=protected-access
    step = torch.compile(
        lambda ids, mask: fwd(model, ids, mask),
        backend=args.compile_backend,
        dynamic=None if args.compile_dynamic == "none" else False,
        options=json.loads(args.compile_options) or None)
  else:
    step = lambda ids, mask: fwd(model, ids, mask)

  times, metrics, widths = [], [], []
  wall = time.perf_counter()
  with torch.inference_mode():
    for (seqs, groups), ref in zip(batches, refs):
      ids, mask = pad(seqs, strategy, args)
      widths.append(ids.shape[1])
      ids, mask = ids.to(device), mask.to(device)
      t = time.perf_counter()
      out = step(ids, mask)
      sync(device)
      times.append(time.perf_counter() - t)
      metrics.append(compare(task, out.cpu(), ref, groups))
  wall = time.perf_counter() - wall

  steady = statistics.median(times[len(times) // 2:])
  slow = [t for t in times if t > 10 * steady]
  n_texts = sum(len(s) for s, _ in batches)
  row = {
      "first_call_s": times[0],
      "compile_like_calls": len(slow),
      "compile_like_total_s": sum(slow),
      "steady_median_ms": steady * 1e3,
      "steady_texts_per_s": len(batches[0][0]) / steady,
      "total_wall_s": wall,
      "end_to_end_texts_per_s": n_texts / wall,
      "distinct_widths": sorted(set(widths)),
  }
  if task == "embed":
    row["min_cos"] = min(m["min_cos"] for m in metrics)
    row["mean_cos"] = statistics.mean(m["mean_cos"] for m in metrics)
  else:
    row["max_abs_logit_diff"] = max(m["max_abs_logit_diff"] for m in metrics)
    row["top1_agree"] = statistics.mean(m["top1_agree"] for m in metrics)
  return row


def main():
  args = parse_args()
  device = torch.device(args.device)
  torch._dynamo.config.recompile_limit = 64  # pylint: disable=protected-access
  if args.device != "cpu":
    torch.set_float32_matmul_precision(args.matmul_precision)
  env = environment(args)
  print(json.dumps(env, indent=2), flush=True)

  with open(args.out, "a", encoding="utf-8") as f:
    f.write(json.dumps({"env": env, "args": vars(args)}) + "\n")
    for task in args.tasks.split(","):
      model_cpu, tok = load(args, task)
      for bs in map(int, args.batch_sizes.split(",")):
        batches = make_inputs(args, task, tok, bs)
        with torch.inference_mode():  # CPU float32 reference, unpadded
          refs = [forward_fn(task)(model_cpu,
                                   *pad(s, "longest", args))
                  for s, _ in batches]
        for mode in args.modes.split(","):
          for strategy in args.strategies.split(","):
            key = {"task": task, "mode": mode, "strategy": strategy,
                   "batch_size": bs}
            try:
              row = run_config(args, task, mode, strategy, batches, refs,
                               model_cpu, device)
            except Exception as e:  # pylint: disable=broad-except
              row = {"error": f"{type(e).__name__}: {e}"[:500],
                     "traceback": traceback.format_exc()}
            row = {**key, **row}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(summarize(row), flush=True)


def summarize(r):
  head = f"{r['task']:6} {r['mode']:7} {r['strategy']:7} bs={r['batch_size']:<3}"
  if "error" in r:
    return f"{head} ERROR {r['error'].splitlines()[0][:120]}"
  acc = (f"min_cos={r['min_cos']:.5f}" if "min_cos" in r else
         f"max|dlogit|={r['max_abs_logit_diff']:.4f} "
         f"top1={r['top1_agree']:.2f}")
  return (f"{head} {acc} first={r['first_call_s']:.2f}s "
          f"compiles~{r['compile_like_calls']} "
          f"steady={r['steady_median_ms']:.1f}ms "
          f"e2e={r['end_to_end_texts_per_s']:.1f}/s "
          f"widths={r['distinct_widths']}")


if __name__ == "__main__":
  main()
