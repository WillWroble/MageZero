# Generation cost per RunPod GPU type (Nov 1)

The same config on each pod type: ver13 (value only) vs offline Standard-MonoB, 200 games (`runpod/configs/norm-MonoB.yml`,
threads and JVM heap scaled to the pod: `THREADS` / `HEAP` in `runpod/generate.sh`). EU-RO-1, secure cloud.

| pod | $/h | guaranteed vCPU / RAM | host CPU | threads | 200 games in | games/h | games per $ | Faeries WR |
|---|---|---|---|---|---|---|---|---|
| RTX PRO 6000 | 2.49 | 16 / 94 GB | EPYC 9555 | 64 | 37 min | 322 | 129 | 31.0% |
| RTX 4090 | 0.89 | 8 / 46 GB | Ryzen 9 7950X | 16 | 76 min | 158 | **178** | 30.0% |
| RTX PRO 4000 | 0.57 | 12 / 31 GB | EPYC 7352 | 24 | 128 min | 93 | 164 | 32.2% |
| RTX 5090 | 1.19 | 16 / 60 GB | Ryzen 9 9950X | 32 | 88 min | 136 | 114 | 30.5% |
| A100 SXM | 1.79 | 32 / 250 GB | EPYC 7763 | 64 | 72 min | 167 | 93 | 30.7% |

Conclusion: the RTX 4090 makes games about 38% cheaper than the PRO 6000, at half the speed per pod; the PRO 4000 is
next (+27%) but 3.5× slower. Win rates agree across pods, so play is the same. Use 4090s (twice the pods for the same
wall-clock time) when cost matters, PRO 6000s when time does. Game speed tracks fast cores more than core count.

Search timeouts: the 60 s timeout almost never cuts a search short. Share of Faeries decisions (with a choice)
below 290 visits: PRO 6000 0.1% (MonoB), 0.1% (gen3 offline Elves), 0.0% (gen3 self-play CawGates); RTX 4090 0.0%
(MonoB). On 4090s, generation runs use `TIMEOUT_MS=120000` anyway (`runpod/generate.sh`) so long games can't start
timing out on the slower pods.
