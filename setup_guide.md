# MageZero Setup Guide

This guide walks you through installing MageZero and training your first deck-local RL agent.

If you just want to run a trained model against your own deck in XMage, see the Play section at the bottom. This guide assumes you're here to train.

---

## 1. System Requirements

### Hardware

| | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8–16 cores |
| RAM | 16 GB | 32–64 GB |
| GPU | — | NVIDIA, 8 GB+ VRAM |
| Disk | 25 GB free | 40 GB+ |

MCTS is CPU-heavy. Training uses the GPU. Both can run on one machine.

### Software

- Windows or Linux (Windows tested, Linux support in progress)
- Java 21+
- Python 3.10+
- PyTorch with CUDA ([install guide](https://pytorch.org/get-started/locally/))
- Git

---

## 2. Install MageZero

### 2.1 Clone the repo

```
git clone https://github.com/WillWroble/MageZero
cd MageZero
```

### 2.2 Download the XMage distribution

Go to the [latest release](https://github.com/WillWroble/MageZero/releases/latest) and download `magezero-xmage-v0.1.0-alpha.zip`. Extract it into the repo root so you end up with an `xmage/` folder containing `lib/`, `mz-xmage.bat`, `mz-xmage.sh`, and `log4j.properties`.

```
MageZero/
├── xmage/          ← from the release zip
│   ├── lib/
│   ├── mz-xmage.bat
│   ├── mz-xmage.sh
│   └── log4j.properties
├── src/
├── configs/
└── ...
```

You do not need to clone or build XMage from source. The release ships a precompiled JAR with all dependencies bundled.

### 2.3 Create a virtual environment

```
python -m venv .venv
```

Windows:
```
.venv\Scripts\activate
```

Linux:
```
source .venv/bin/activate
```

### 2.4 Install MageZero

```
pip install -e .
```

This installs the `mz` command globally in your venv and pulls all Python dependencies from `pyproject.toml`.

Verify it worked:
```
mz --help
```

You should see the list of subcommands: `train`, `batch`, `play`, `import`, `export`.

### 2.5 GPU check

```
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If `False`, install the CUDA-matched PyTorch wheel from the link above — the default pip install often gives you the CPU-only build.

---

## 3. Add a Deck

Training is deck-local. You need at least one deck to train and one or more decks for your agent to train against.

### 3.1 Convert your deck to .dck format

Export your deck from Moxfield, MTGA, or MTGO as plain text. Use the XMage client's deck builder (or any `.txt` → `.dck` converter) to produce a `.dck` file.

Sideboards are not yet supported.

### 3.2 Drop it in

```
MageZero/xmage/decks/MyDeck.dck
```

The filename (without `.dck`) is what you'll reference everywhere else — in configs, on the command line, in the data/model folder names. Pick something without spaces.

A handful of reference decks are included in `xmage/decks/` out of the box for opponents — monocolored Standard decks. You can add your own or use those.

---

## 4. Configure Your Run

Open `configs/run.yml`. This is the main file you'll edit.

```yaml
deck: MyDeck
version: 1
start_from_version: null

generations: 6
games_per_gen: 200
replay_buffer_gens: 3

opponents:
  - deck: Standard-MonoR
    mode: minimax
  - deck: Standard-MonoG
    mode: minimax

training:
  analyze_dataset: true
  generate_plots: true
  eval_previous_model: true

log_level: ACTIONS
curriculum: configs/curriculum.yml
```

Fields worth knowing:

- **`deck`** — the deck you're training. Must match a file at `xmage/decks/<deck>.dck`.
- **`version`** — the version number for this training run. Defaults to 1. Increment if you want to start fresh without overwriting a previous run's models and data.
- **`start_from_version`** — set to a previous version number to start training from that checkpoint instead of offline bootstrap. Leave as `null` for a first-time run.
- **`generations`** — how many full self-play / train / eval cycles to run.
- **`games_per_gen`** — how many games per opponent per generation.
- **`replay_buffer_gens`** — how many past generations of data to train on. Older data gets archived automatically.
- **`opponents`** — list of decks to play against. `mode` is `minimax` (fast, heuristic bot) or `mcts` (full MCTS with its own trained model). Start with `minimax` for initial training.

### Curriculum (optional)

`configs/curriculum.yml` controls hyperparameter annealing across generations. The defaults are reasonable — you don't need to touch this unless you want to tune the schedule. See comments in the file for details.

### Game parameters (optional)

`configs/game.yml` controls XMage-side parameters the runner passes through (search budget, threads, MCTS timeout, hidden info toggle, etc.). Edit if you want to tweak the engine, but the defaults work.

---

## 5. Train

```
mz train
```

That's the whole command. `mz train` will:

1. Create a run folder under `runs/<timestamp>/`
2. For each generation:
   - Launch the primary Python inference server (or run offline for gen 0)
   - For each opponent: start/stop its server if needed, launch XMage via the JAR, generate games
   - Run dataset stats on the new data (saves plots)
   - Evaluate the previous generation's model on the new data (gen 1+)
   - Move new data from `testing/` to `training/`
   - Archive out-of-window training data
   - Train the next model (50 epochs bootstrap, 10 epochs from checkpoint)
3. Mark the run as complete

The terminal shows XMage game logs in real time. Python-side output (training loss, dataset stats, test accuracy) is captured in per-run log files instead of flooding your terminal.

### Resume an interrupted run

If `mz train` is interrupted (Ctrl+C, crash, power outage), the next time you run it it'll detect the incomplete run and ask:

```
Active run found: 2026-04-09_14-23-42. Resume? [Y/n]
```

Answer `Y` to pick up where you left off. Answer `n` to abandon it and start fresh — the abandoned run is marked in its `run.json` but never deleted, so the history is preserved.

### What gets produced

```
data/MyDeck/ver1/
├── testing/                 ← new data arrives here first
├── training/                ← active training buffer
└── archive/                 ← out-of-window data

models/MyDeck/ver1/
├── model.pt.gz              ← latest checkpoint
├── ignore.roar              ← feature ignore list
└── state.json               ← deck session counter

plots/MyDeck/ver1/
├── value_hist_*.png
├── avg_policy_*.png
└── idx_dist_*.png

runs/2026-04-09_14-23-42/
├── run.json                 ← full run history
├── train.log                ← all gens, appended
├── test.log
├── dataset_stats.log
└── server_50052.log
```

Every run is fully self-contained under `runs/`. You can grep across a run's full training history with `grep foo runs/2026-04-09_14-23-42/train.log`.

---

## 6. Monitor a Run

Training can take hours. Watch progress live from another terminal:

```
# Tail the training output
Get-Content runs\<run_id>\train.log -Wait -Tail 20

# Watch data files appear
dir data\MyDeck\ver1\testing
```

The `run.json` file is updated after every gen with the current stage (`generate`, `analyze`, `eval`, `move`, `train`) — check it to see what the runner is doing right now.

---

## 7. Other Commands

### `mz batch`
Launch a single XMage data generation run using `configs/game.yml` directly. No orchestration, no training — just XMage. Useful for manual experiments outside the curriculum pipeline.

```
mz batch --config configs/game.yml
```

### `mz import <file>`
Auto-detects file type:
- `.dck` → copies to `xmage/decks/`
- `.mz` → unpacks to `models/<deck>/ver<N>/`
- `.txt` → not yet wired up, convert manually for now

```
mz import path/to/mydeck.dck
```

### `mz export`
Pack a trained model into a shareable `.mz` bundle.

```
mz export --deck MyDeck --version 3
```

Produces `exports/MyDeck_v3.mz` containing the model, ignore list, and metadata.

### `mz play`
Host a local game server with a trained AI player. Starts the inference server for the chosen opponent deck, launches XMage, and waits for you to connect with the XMage client. See the Play section below for full instructions.

```
mz play --deck UWTempo
mz play --deck UWTempo --version 3
mz play --deck UWTempo --config configs/game.yml

```

---
## 8. Play Against a Trained Model

If you don't want to train — just play — you still need the base install (steps 2.1–2.4). You also need a trained model: either train one yourself or import a `.mz` bundle someone shared with you via `mz import model.mz`.

### Start the server

`mz play --deck UWTempo`

This starts the Python inference server and launches an XMage game server. `--version` is optional; it defaults to the latest trained version for that deck. `--config` points to a `game.yml` file (defaults to `configs/game.yml`) which controls MCTS parameters like search budget, timeout, and prior temperature during play.

### Connect with the XMage client

1. Open the vanilla [XMage client](https://xmage.today/)
2. Connect to `localhost` (default port)
3. Create a new match and select the **MCTS** bot type as your opponent
4. Choose any deck you want to play but make sure to give the opponent the same deck you provided in the `play` command
5. Start the game

## 9. Troubleshooting

**`mz train` hangs at "Loading database..."** — First launch builds the H2 card database from scratch. Takes ~1 minute. Subsequent runs skip this.

**`HDF5FileNotFoundException: Directory does not exist`** — The runner creates parent directories automatically, but `mz batch` with a custom `output_file` does not. Create the parent folder manually or use `mz train` instead.

**`Failed to establish connection to network model`** — The Java side couldn't reach the Python server. Check that `server_50052.log` in your run folder shows the server started successfully. If the server failed, runner would have aborted — so this warning is usually benign and means the JVM launched before the server was fully ready. Not a problem, servers have a 600s warmup window.

**`java.lang.OutOfMemoryError`** — Lower `games_per_gen` or reduce `MAX_CONCURRENT_GAMES` in the XMage source. Default heap is `-Xmx24g` in `mz-xmage.bat`; bump it up if you have more RAM.

**Couldn't load deck, deck size=0** — Card set jar missing or corrupt. Redownload the XMage distribution zip from the release page and re-extract.

**ZGC not supported** — Edit `xmage/mz-xmage.bat` and remove `-XX:+UseZGC`. G1GC (the default) will be used instead, slightly slower but works everywhere.

---

## 10. Contact

- Discord: `inkling_6`
- Discord server: [https://discord.gg/R6pB6xuEy9](https://discord.gg/R6pB6xuEy9)
- Email: `willwroble@gmail.com`

Bug reports, feature requests, and "how do I..." questions are all welcome.