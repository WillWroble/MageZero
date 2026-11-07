# Policy only: ver15's policy without search

Faeries is `on_policy` (`ComputerPlayerPolicyOnly`): at every decision it plays the network's top-scored legal move
(priority, target and choose-use priors on, no noise), with no search. The opponent is the offline heuristic search
(300 sims). 200 games per deck. Kaggle T4, 4 vCPUs, 8 game threads (`kaggle/generate.py`, configs in
`kaggle/configs/onpolicy-<deck>.yml`). Logs in `results/kaggle/onpolicy-<deck>/`.

Reference: the same model with value-only search (300 sims, priors off), from `results/gen3-training.md`.

| deck | XMage commit | policy only | ver15 with search |
|---|---|---|---|
| MonoB | `a800f4fe` (priors.opponent true) | 12.5% (25/200) | 36.0% |

- MonoB: the run needed `priors.opponent: true`. With it false, every game failed at `a800f4fe` on a null
  `node.policy` in `bestAction` (logs in the first commit of `results/kaggle/onpolicy-monob`). Faeries plays its
  spells (Humbling Elder, Spellstutter, ninjutsu, Of One Mind and Strix loot appear in the game log), so the result
  isn't the player passing. One error line in the game log.
