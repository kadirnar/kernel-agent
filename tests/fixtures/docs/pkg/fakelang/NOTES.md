# fakelang notes

## Pipelining

Loops over K are software pipelined with `num_stages` buffers in shared memory; the
compiler inserts the waits.

```python
# a comment in a code block, not a heading
for k in range(0, K, BK):
    acc = dot(a, b, acc)
```

## Tiny

short
