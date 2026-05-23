"""
cli.py — `mz` command entry point.

Commands:
  mz train                          full curriculum pipeline (auto-resume)
  mz batch [--config FILE]          single JVM launch via game.yml
  mz play  --deck X [--version N]   host a local AI player (stub)
  mz import <file>                  auto-detects .dck or .mz (.txt stubbed)
  mz export --deck X --version N    pack model into a .mz bundle
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from magezero.util.config import load_all
from magezero import runner


# ─── train ───────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    run_cfg, cur_cfg = load_all(args.run)
    runner.run_pipeline(run_cfg, cur_cfg, base_game_yml=args.game)


# ─── batch ───────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> None:
    runner.launch_jvm(args.config)


# ─── play ────────────────────────────────────────────────────

def cmd_play(args: argparse.Namespace) -> None:
    import subprocess
    from pathlib import Path

    config = Path(args.config).resolve()
    if not config.exists():
        sys.exit(f"config not found: {config}")

    deck = args.deck
    version = args.version
    if version is None:
        version = runner.latest_version(deck)
    server = None
    if version is None or not runner.has_checkpoint(deck, version):
    #if not runner.has_checkpoint(deck, version):
        print(f"model for {deck} is not found, falling back to offline MCTS")
        #sys.exit(f"no checkpoint at models/{deck}/ver{version}/model.pt.gz")
    else:
        # start inference server
        print(f"[play] starting inference server for {deck} v{version}")
        server = runner.start_server(deck, version, runner.PRIMARY_PORT, Path("."))

    try:
        # launch XMage server
        script = "xmage\\mz-xmage-play.bat" if sys.platform == "win32" else "xmage/mz-xmage-play.sh"
        cmd = ["cmd", "/c", script, str(config)] if sys.platform == "win32" else [script, str(config)]
        subprocess.run(cmd, check=True)
    finally:
        if server is not None:
            runner.stop_server(server)


# ─── import ──────────────────────────────────────────────────

def cmd_import(args: argparse.Namespace) -> None:
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"file not found: {src}")

    suffix = src.suffix.lower()
    if suffix == ".dck":
        dst = Path("xmage/decks") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"✓ imported deck → {dst}")

    elif suffix == ".mz":
        with zipfile.ZipFile(src) as zf:
            meta = json.loads(zf.read("metadata.json"))
            deck = meta["deck"]
            version = meta["version"]
            dst = Path("models") / deck / f"ver{version}"
            dst.mkdir(parents=True, exist_ok=True)
            zf.extract("model.pt.gz", dst)
            zf.extract("ignore.roar", dst)
        print(f"✓ imported model → {dst}")

    elif suffix == ".txt":
        sys.exit("`.txt` deck conversion not yet wired up. "
                 "Convert manually to .dck for now.")

    else:
        sys.exit(f"unknown file type: {suffix} (expected .dck, .mz, or .txt)")


# ─── export ──────────────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> None:
    src = Path("models") / args.deck / f"ver{args.version}"
    if not src.exists():
        sys.exit(f"model not found: {src}")

    model_file = src / "model.pt.gz"
    ignore_file = src / "ignore.roar"
    if not model_file.exists() or not ignore_file.exists():
        sys.exit(f"missing model.pt.gz or ignore.roar in {src}")

    out_dir = Path("exports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.deck}_v{args.version}.mz"

    metadata = {"deck": args.deck, "version": args.version}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_file, "model.pt.gz")
        zf.write(ignore_file, "ignore.roar")
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    print(f"✓ exported → {out_path}")


# ─── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="mz")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="full curriculum pipeline")
    p_train.add_argument("--run", default="configs/run.yml")
    p_train.add_argument("--game", default="configs/game.yml")
    p_train.set_defaults(func=cmd_train)

    p_batch = sub.add_parser("batch", help="single JVM launch")
    p_batch.add_argument("--config", default="configs/game.yml")
    p_batch.set_defaults(func=cmd_batch)

    p_play = sub.add_parser("play", help="host a local AI player")
    p_play.add_argument("--deck", required=True)
    p_play.add_argument("--version", type=int, default=None)
    p_play.add_argument("--config", default="configs/game.yml")
    p_play.set_defaults(func=cmd_play)

    p_import = sub.add_parser("import", help="import .dck or .mz file")
    p_import.add_argument("file")
    p_import.set_defaults(func=cmd_import)

    p_export = sub.add_parser("export", help="export model as .mz bundle")
    p_export.add_argument("--deck", required=True)
    p_export.add_argument("--version", type=int, required=True)
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()