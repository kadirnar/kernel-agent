### S1. Agent evaluations per hour vs k (cell: evals/h, GPU busy, mean wait of an evaluation)

† timed GPU work overlapped another agent's unlocked dev run (> 5% of timed GPU time: timing not clean). ‡ integration falls behind (backlog > 1 h of A/B work after 400 simulated hours: not sustainable).

**pooled (all 4 runs)** (bootstrap pool: 86 evaluating-arm sessions; integration GPU-s per agent eval: V-lat-r1 103, V-lat 117, V-thr 352, Q-lat 111)

| scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| A: no integration; dev runs unlocked (as today) | 8.2 (12%, 0s) | 16.2 (23%, 9s)† | 23.8 (34%, 18s)† | 30.9 (44%, 29s)† | 43.8 (62%, 57s)† |
| B: integration at today's ratio, one FIFO queue; dev unlocked | 7.2 (48%, 64s)† | 11.9 (79%, 166s)† | 14.2 (95%, 322s)† | 14.9 (99%, 529s)† | 15.0 (100%, 1002s)† |
| C: integration at today's ratio, priority eval > integration; dev unlocked | 7.5 (50%, 48s)† | 13.6 (91%, 96s)† | 19.7 (100%, 112s)†‡ | 26.2 (100%, 113s)†‡ | 37.9 (100%, 134s)†‡ |
| D: clean: dev runs exclusive too (f=0.5), priority eval > dev > integration | 6.4 (59%, 21s) | 9.4 (87%, 44s) | 10.8 (100%, 64s)‡ | 14.4 (100%, 54s)‡ | 20.5 (100%, 40s)‡ |
| E: clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest | 6.4 (59%, 21s) | 9.6 (85%, 42s) | 11.6 (97%, 64s) | 13.6 (100%, 73s)‡ | 19.3 (100%, 72s)‡ |
| F: as E, integration 4x cheaper | 7.7 (41%, 5s) | 13.5 (67%, 19s) | 17.7 (83%, 30s) | 20.7 (92%, 41s) | 24.5 (100%, 62s)‡ |
| G: as D, integration 4x cheaper | 7.7 (41%, 5s) | 12.9 (69%, 17s) | 16.2 (86%, 26s) | 18.0 (96%, 34s) | 21.8 (100%, 35s)‡ |
| H: as E, no integration | 8.2 (33%, 0s) | 15.2 (57%, 13s) | 20.8 (73%, 23s) | 25.3 (84%, 32s) | 31.8 (94%, 47s) |
| D1: as G with f=1.0 (every dev second on the GPU) | 7.7 (60%, 5s) | 11.1 (88%, 33s) | 12.5 (98%, 50s) | 13.5 (100%, 51s)‡ | 15.0 (100%, 49s)‡ |
| D2: as G with f=0.25 | 7.8 (31%, 5s) | 13.6 (55%, 12s) | 17.9 (72%, 19s) | 21.1 (85%, 25s) | 24.5 (99%, 35s) |

**V-lat (latency)** (bootstrap pool: 34 evaluating-arm sessions; integration GPU-s per agent eval: V-lat 117)

| scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| A: no integration; dev runs unlocked (as today) | 5.8 (4%, 0s) | 11.8 (8%, 1s)† | 17.7 (12%, 2s)† | 23.6 (16%, 4s)† | 35.2 (24%, 7s)† |
| B: integration at today's ratio, one FIFO queue; dev unlocked | 5.8 (23%, 19s)† | 11.2 (44%, 39s)† | 16.1 (63%, 64s)† | 20.3 (79%, 103s)† | 24.9 (98%, 259s)† |
| C: integration at today's ratio, priority eval > integration; dev unlocked | 5.8 (23%, 16s)† | 11.3 (44%, 30s)† | 16.5 (65%, 46s)† | 21.4 (84%, 62s)† | 31.4 (100%, 80s)†‡ |
| D: clean: dev runs exclusive too (f=0.5), priority eval > dev > integration | 5.4 (44%, 10s) | 8.8 (72%, 24s) | 10.9 (89%, 37s) | 11.9 (98%, 47s) | 16.0 (100%, 40s)‡ |
| E: clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest | 5.4 (44%, 10s) | 9.3 (71%, 29s) | 12.1 (86%, 44s) | 14.0 (94%, 58s) | 16.3 (100%, 82s)‡ |
| F: as E, integration 4x cheaper | 5.8 (33%, 3s) | 10.8 (57%, 18s) | 15.0 (73%, 30s) | 18.4 (83%, 41s) | 23.4 (94%, 57s) |
| G: as D, integration 4x cheaper | 5.8 (33%, 3s) | 10.2 (59%, 11s) | 13.3 (77%, 19s) | 15.2 (88%, 27s) | 17.1 (99%, 34s) |
| H: as E, no integration | 5.8 (29%, 0s) | 11.4 (51%, 15s) | 16.4 (67%, 26s) | 20.4 (78%, 36s) | 27.1 (91%, 51s) |

**V-thr (throughput)** (bootstrap pool: 27 evaluating-arm sessions; integration GPU-s per agent eval: V-thr 352)

| scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| A: no integration; dev runs unlocked (as today) | 12.1 (19%, 0s) | 23.7 (36%, 9s)† | 34.2 (52%, 20s)† | 43.5 (67%, 36s)† | 56.4 (87%, 87s)† |
| B: integration at today's ratio, one FIFO queue; dev unlocked | 7.6 (85%, 180s)† | 8.8 (100%, 521s)† | 8.8 (100%, 926s)† | 8.9 (100%, 1330s)† | 8.9 (100%, 2145s)† |
| C: integration at today's ratio, priority eval > integration; dev unlocked | 8.3 (94%, 139s)† | 16.0 (100%, 155s)†‡ | 24.1 (100%, 154s)†‡ | 31.9 (100%, 155s)†‡ | 46.7 (100%, 166s)†‡ |
| D: clean: dev runs exclusive too (f=0.5), priority eval > dev > integration | 6.2 (78%, 39s) | 7.6 (96%, 108s) | 9.6 (100%, 124s)‡ | 13.7 (100%, 96s)‡ | 22.6 (100%, 58s)‡ |
| E: clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest | 6.2 (78%, 39s) | 7.6 (94%, 75s) | 8.2 (100%, 120s) | 10.1 (100%, 115s)‡ | 14.5 (100%, 92s)‡ |
| F: as E, integration 4x cheaper | 9.8 (52%, 11s) | 14.8 (77%, 20s) | 17.7 (89%, 31s) | 19.5 (95%, 42s) | 21.5 (99%, 70s) |
| G: as D, integration 4x cheaper | 9.8 (52%, 11s) | 14.7 (78%, 29s) | 17.0 (91%, 44s) | 18.1 (97%, 59s) | 22.6 (100%, 59s)‡ |
| H: as E, no integration | 12.1 (36%, 0s) | 21.8 (61%, 8s) | 29.1 (77%, 15s) | 34.5 (86%, 23s) | 41.8 (95%, 36s) |

**Q-lat (Qwen3)** (bootstrap pool: 9 evaluating-arm sessions; integration GPU-s per agent eval: Q-lat 111)

| scenario | k=1 | k=2 | k=3 | k=4 | k=6 |
|---|---:|---:|---:|---:|---:|
| A: no integration; dev runs unlocked (as today) | 8.7 (11%, 0s) | 16.9 (21%, 7s)† | 25.0 (31%, 15s)† | 32.7 (41%, 24s)† | 46.6 (58%, 47s)† |
| B: integration at today's ratio, one FIFO queue; dev unlocked | 8.3 (36%, 13s)† | 15.6 (67%, 46s)† | 20.6 (89%, 107s)† | 22.7 (99%, 217s)† | 23.2 (100%, 516s)† |
| C: integration at today's ratio, priority eval > integration; dev unlocked | 8.5 (37%, 5s)† | 16.5 (72%, 18s)† | 24.1 (100%, 33s)†‡ | 31.2 (100%, 43s)†‡ | 45.2 (100%, 64s)†‡ |
| D: clean: dev runs exclusive too (f=0.5), priority eval > dev > integration | 8.0 (55%, 6s) | 13.5 (91%, 21s) | 17.6 (100%, 25s)‡ | 21.2 (100%, 26s)‡ | 25.3 (100%, 27s)‡ |
| E: clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest | 8.0 (55%, 6s) | 13.8 (88%, 20s) | 18.5 (100%, 29s)‡ | 23.5 (100%, 34s)‡ | 30.8 (100%, 45s)‡ |
| F: as E, integration 4x cheaper | 8.6 (38%, 2s) | 15.4 (64%, 11s) | 21.1 (81%, 19s) | 25.5 (91%, 28s) | 31.7 (100%, 44s) |
| G: as D, integration 4x cheaper | 8.6 (38%, 2s) | 15.0 (67%, 13s) | 19.2 (86%, 20s) | 21.7 (97%, 25s) | 25.2 (100%, 27s)‡ |
| H: as E, no integration | 8.7 (32%, 0s) | 16.1 (56%, 9s) | 22.6 (72%, 17s) | 28.0 (83%, 25s) | 36.5 (94%, 38s) |

### S2. Pooled detail for the clean-timing scenarios

**E: clean: reader/writer lock (timed exclusive, dev shared, f=0.5), integration lowest**

| k | evals/h | vs k=1 | per-agent efficiency | GPU eval / integration / dev | GPU busy | eval wait mean / p90 s | dev wait s | agent time blocked | integration backlog h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.4 | 1.00x | 100% | 9% / 34% / 17% | 59% | 21 / 61 | 21 | 22% | -0.0 |
| 2 | 9.6 | 1.49x | 75% | 14% / 51% / 21% | 85% | 42 / 135 | 53 | 41% | 0.0 |
| 3 | 11.6 | 1.80x | 60% | 16% / 60% / 21% | 97% | 64 / 183 | 87 | 54% | 0.0 |
| 4 | 13.6 | 2.12x | 53% | 20% / 60% / 21% | 100% | 73 / 197 | 109 | 59% | 44.6 |
| 6 | 19.3 | 3.01x | 50% | 27% / 48% / 25% | 100% | 72 / 172 | 120 | 61% | 211.3 |

**F: as E, integration 4x cheaper**

| k | evals/h | vs k=1 | per-agent efficiency | GPU eval / integration / dev | GPU busy | eval wait mean / p90 s | dev wait s | agent time blocked | integration backlog h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 | 1.00x | 100% | 11% / 10% / 20% | 41% | 5 / 0 | 5 | 6% | -0.0 |
| 2 | 13.5 | 1.74x | 87% | 19% / 18% / 31% | 67% | 19 / 55 | 16 | 18% | -0.0 |
| 3 | 17.7 | 2.28x | 76% | 25% / 23% / 35% | 83% | 30 / 79 | 28 | 28% | 0.0 |
| 4 | 20.7 | 2.68x | 67% | 29% / 27% / 36% | 92% | 41 / 98 | 43 | 37% | 0.0 |
| 6 | 24.5 | 3.17x | 53% | 35% / 32% / 33% | 100% | 62 / 136 | 75 | 50% | 1.8 |

**G: as D, integration 4x cheaper**

| k | evals/h | vs k=1 | per-agent efficiency | GPU eval / integration / dev | GPU busy | eval wait mean / p90 s | dev wait s | agent time blocked | integration backlog h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 | 1.00x | 100% | 11% / 10% / 20% | 41% | 5 / 0 | 5 | 6% | -0.0 |
| 2 | 12.9 | 1.67x | 83% | 18% / 17% / 34% | 69% | 17 / 53 | 21 | 22% | 0.0 |
| 3 | 16.2 | 2.09x | 70% | 23% / 21% / 42% | 86% | 26 / 71 | 40 | 34% | 0.0 |
| 4 | 18.0 | 2.33x | 58% | 25% / 24% / 47% | 96% | 34 / 83 | 65 | 45% | 0.0 |
| 6 | 21.8 | 2.82x | 47% | 30% / 13% / 56% | 100% | 35 / 82 | 103 | 56% | 61.1 |

**H: as E, no integration**

| k | evals/h | vs k=1 | per-agent efficiency | GPU eval / integration / dev | GPU busy | eval wait mean / p90 s | dev wait s | agent time blocked | integration backlog h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.2 | 1.00x | 100% | 12% / 0% / 21% | 33% | 0 / 0 | 0 | 0% | 0.0 |
| 2 | 15.2 | 1.84x | 92% | 22% / 0% / 35% | 57% | 13 / 29 | 5 | 8% | 0.0 |
| 3 | 20.8 | 2.52x | 84% | 30% / 0% / 43% | 73% | 23 / 61 | 12 | 16% | 0.0 |
| 4 | 25.3 | 3.07x | 77% | 36% / 0% / 48% | 84% | 32 / 78 | 20 | 23% | 0.0 |
| 6 | 31.8 | 3.86x | 64% | 45% / 0% / 49% | 94% | 47 / 102 | 39 | 36% | 0.0 |
