<!-- One problem per issue. Search open and closed issues first. Cut the repro
down until removing anything else makes the bug disappear. -->

**Title:** `<op or feature>`: `<what goes wrong>` `<under which condition>`
(example: "XLM-R encoder: torch.compile recompiles on every batch when the
attention mask's seq dim changes")

### What happened vs. what I expected

Expected: ...
Actual: ...

### Minimal reproducible example

```python
import torch

device = torch.device("tpu")
# no transformers import if it can be avoided; inline the 2-3 ops involved
```

Output / full traceback:

```text
```

### Environment

- TPU: v5e-1 (Colab) / v6e-1
- Python:
- torch:
- torch_tpu:
- libtpu:
- transformers (only if needed for the repro):
- Mode: eager / `torch.compile(backend="tpu", dynamic=None)`
- `torch.get_float32_matmul_precision()`:

### Notes

Related issues: #...
What I already ruled out (e.g. "same result with
`torch.set_float32_matmul_precision('highest')`"): ...
