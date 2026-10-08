### `median` (seeds 0-4, mean)

| --agents | evaluations | per h | 5-hour window max | account spent | sessions stopped at the limit | final speedup | mean k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 8.8 | 46% | 0.00 h | 0.0 | 2.41x | — |
| 2 | 118 | 13.7 | 68% | 0.00 h | 0.0 | 2.52x | — |
| 3 | 137 | 16.2 | 80% | 0.00 h | 0.0 | 2.53x | — |
| 4 | 152 | 18.3 | 87% | 0.00 h | 0.0 | 2.52x | — |
| auto | 121 | 14.2 | 71% | 0.00 h | 0.0 | 2.52x | 3.4 |
| 3 +async | 139 | 16.5 | 96% | 0.00 h | 0.4 | 2.52x | — |
| auto +async | 122 | 14.0 | 83% | 0.00 h | 0.0 | 2.53x | 2.4 |

### `used40` (seeds 0-4, mean)

| --agents | evaluations | per h | 5-hour window max | account spent | sessions stopped at the limit | final speedup | mean k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 8.8 | 76% | 0.00 h | 0.0 | 2.41x | — |
| 2 | 118 | 13.5 | 98% | 0.10 h | 0.4 | 2.52x | — |
| 3 | 125 | 14.6 | 100% | 0.63 h | 1.6 | 2.52x | — |
| 4 | 135 | 16.0 | 100% | 1.35 h | 1.8 | 2.52x | — |
| auto | 118 | 12.9 | 92% | 0.00 h | 0.0 | 2.52x | 2.6 |
| 3 +async | 118 | 14.3 | 100% | 1.79 h | 2.8 | 2.52x | — |
| auto +async | 117 | 13.6 | 95% | 0.00 h | 0.0 | 2.52x | 2.0 |

### `p90` (seeds 0-4, mean)

| --agents | evaluations | per h | 5-hour window max | account spent | sessions stopped at the limit | final speedup | mean k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 8.8 | 74% | 0.00 h | 0.0 | 2.41x | — |
| 2 | 106 | 13.2 | 99% | 0.74 h | 3.0 | 2.50x | — |
| 3 | 106 | 13.4 | 100% | 1.46 h | 4.4 | 2.51x | — |
| 4 | 106 | 14.1 | 100% | 2.05 h | 3.8 | 2.52x | — |
| auto | 102 | 12.0 | 95% | 0.00 h | 0.0 | 2.50x | 1.5 |
| 3 +async | 94 | 13.0 | 100% | 2.25 h | 5.0 | 2.49x | — |
| auto +async | 88 | 10.5 | 99% | 0.03 h | 0.4 | 2.47x | 1.2 |
