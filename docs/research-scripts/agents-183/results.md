### `--dry-run --max-hours 8` (seeds 0-4, mean)

| --agents | evaluations / simulated h | GPU busy | eval wait mean / p95 | evaluations | loop h | done at h | final speedup | rounds | $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (sequential) | 7.0 | 43% | 0 s / 0 s | 47 | 6.78 | 6.99 | 2.357x | 1.0 | 23 |
| 1 (coordinator) | 9.3 | 51% | 33 s / 150 s | 63 | 6.80 | 7.29 | 2.375x | 1.4 | 29 |
| 2 | 14.8 | 72% | 51 s / 165 s | 99 | 6.72 | 7.44 | 2.500x | 2.2 | 47 |
| 3 | 17.0 | 84% | 66 s / 184 s | 118 | 6.95 | 7.29 | 2.516x | 2.6 | 58 |
| 4 | 18.5 | 89% | 70 s / 195 s | 121 | 6.55 | 7.16 | 2.520x | 2.4 | 59 |

### `--dry-run --rounds 2` (seeds 0-4, mean)

| --agents | evaluations / simulated h | GPU busy | eval wait mean / p95 | evaluations | loop h | done at h | final speedup | rounds | $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (sequential) | 7.3 | 39% | 0 s / 0 s | 120 | 16.25 | 16.78 | 2.529x | 2.0 | 58 |
| 1 (coordinator) | 8.9 | 51% | 32 s / 151 s | 118 | 13.16 | 13.79 | 2.526x | 2.0 | 57 |
| 2 | 14.1 | 77% | 55 s / 169 s | 115 | 8.13 | 8.48 | 2.519x | 2.0 | 56 |
| 3 | 17.1 | 88% | 74 s / 192 s | 116 | 6.72 | 7.13 | 2.525x | 2.0 | 57 |
| 4 | 18.2 | 94% | 76 s / 199 s | 117 | 6.35 | 6.64 | 2.523x | 2.0 | 56 |
