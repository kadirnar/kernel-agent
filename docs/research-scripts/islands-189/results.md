### `--dry-run --agents 3 --max-hours 8` (seeds 0-4, mean)

| --islands | kernel evaluations | final speedup | agent h | log gain / agent h | loop h | rounds | migrations offered / used | reseeded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (default) | 100 | 2.516x | 13.6 | 0.0678 | 6.95 | 2.6 | 0.0 / 0.0 | 0.0 |
| 2 | 117 | 2.686x | 15.2 | 0.0652 | 6.71 | 2.2 | 6.6 / 2.6 | 0.0 |
| 2, independent | 117 | 2.696x | 15.2 | 0.0652 | 6.72 | 2.2 | 0.0 / 0.0 | 0.0 |
| 3 | 127 | 2.611x | 16.6 | 0.0582 | 7.04 | 2.6 | 6.0 / 3.6 | 0.0 |
| 3, independent | 126 | 2.633x | 16.6 | 0.0589 | 6.88 | 2.4 | 0.0 / 0.0 | 0.0 |

### `--dry-run --agents 3 --rounds 2` (seeds 0-4, mean)

| --islands | kernel evaluations | final speedup | agent h | log gain / agent h | loop h | rounds | migrations offered / used | reseeded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (default) | 99 | 2.525x | 13.6 | 0.0692 | 6.72 | 2.0 | 0.0 / 0.0 | 0.0 |
| 2 | 118 | 2.695x | 15.8 | 0.0631 | 7.43 | 2.0 | 6.0 / 2.4 | 0.0 |
| 2, independent | 123 | 2.689x | 16.4 | 0.0607 | 7.55 | 2.0 | 0.0 / 0.0 | 0.0 |
| 3 | 136 | 2.623x | 17.9 | 0.0545 | 7.53 | 2.0 | 7.4 / 4.6 | 0.0 |
| 3, independent | 133 | 2.641x | 17.5 | 0.0559 | 7.39 | 2.0 | 0.0 / 0.0 | 0.0 |
