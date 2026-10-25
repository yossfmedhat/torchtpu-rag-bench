"""Find the smallest graph that fails to lower with a dynamic seq dim on TPU.

Seen with XLM-R (BGE-M3) under torch.compile(backend=BACKEND, dynamic=None):
  'stablehlo.broadcast_in_dim' op result #0 must be statically shaped or
  single bounded dimension tensor ... but got 'tensor<8x?x1024xf32>'
at the embeddings LayerNorm, once Dynamo marks the seq dim dynamic.

Each case runs three seq lengths (the 2nd call makes Dynamo mark dim 1
dynamic) and prints OK or the first line of the error.
"""

import os

import torch
from torch import nn

DEVICE = torch.device(os.environ.get("DEVICE", "tpu"))
BACKEND = os.environ.get("BACKEND", "tpu")
B, H, VOCAB = 8, 1024, 1000
LENGTHS = (115, 124, 250)


class AddNorm(nn.Module):

  def __init__(self):
    super().__init__()
    self.norm = nn.LayerNorm(H)

  def forward(self, x):
    return self.norm(x + x)


class PositionOnly(nn.Module):
  """Position embedding from arange(seq_len), expanded to the batch."""

  def __init__(self):
    super().__init__()
    self.pos = nn.Embedding(514, H)

  def forward(self, ids):
    pos_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    return self.pos(pos_ids).expand(ids.shape[0], -1, -1)


class Embeddings(nn.Module):
  """word + position + token_type embeddings, then LayerNorm (XLM-R style)."""

  def __init__(self, layer_norm=True):
    super().__init__()
    self.word = nn.Embedding(VOCAB, H)
    self.pos = nn.Embedding(514, H)
    self.tok = nn.Embedding(1, H)
    self.norm = nn.LayerNorm(H) if layer_norm else nn.Identity()

  def forward(self, ids):
    pos_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    tok_ids = torch.zeros_like(ids)
    return self.norm(self.word(ids) + self.pos(pos_ids) + self.tok(tok_ids))


def hidden(s):
  return torch.randn(B, s, H, device=DEVICE)


def ids(s):
  return torch.randint(0, VOCAB, (B, s), device=DEVICE)


def run(name, module, make_input, mark=None):
  torch._dynamo.reset()  # pylint: disable=protected-access
  module = module.to(DEVICE).eval()
  step = torch.compile(module, backend=BACKEND, dynamic=None)
  try:
    with torch.inference_mode():
      for s in LENGTHS:
        x = make_input(s)
        if mark:
          torch._dynamo.mark_dynamic(x, 1, **mark)  # pylint: disable=protected-access
        out = step(x)
        if DEVICE.type != "cpu":
          torch.accelerator.synchronize()
        assert out.shape[1] == s, out.shape
    print(f"OK    {name}")
  except Exception as e:  # pylint: disable=broad-except
    msg = str(e).strip().splitlines()[0][:160]
    print(f"FAIL  {name}: {type(e).__name__}: {msg}")


if __name__ == "__main__":
  print("torch", torch.__version__)
  run("1 LayerNorm only", nn.LayerNorm(H), hidden)
  run("2 add + LayerNorm", AddNorm(), hidden)
  run("3 position embedding only", PositionOnly(), ids)
  run("4 embeddings, no LayerNorm", Embeddings(layer_norm=False), ids)
  run("5 embeddings + LayerNorm", Embeddings(), ids)
  run("6 embeddings + LayerNorm, mark_dynamic(min=2, max=512)", Embeddings(),
      ids, mark={"min": 2, "max": 512})
