# Experiment data sets under /workspace/data, from the repo's shards (run by files.sh):
#   gen0/train  ver4 gen0, Pauper pool (8 matchups incl. the mirror), both seats (the set Dama compared on)
#   gen0/test   ver4 gen0 mirror test set
#   gen1/train, gen1/test  every gen1 shard (ver4, ver5) except the hidden-information _pimc ones, split by game:
#               the last 20% of each shard's games are held out (built once, before the gen1 folders were reorganized)
#   both/train  gen0/train and gen1/train
#   gen0-nomirror/train  gen0/train without the mirror shards: the mirror is then an unseen opponent
#   gen0-mono/test       the first 100 games of each gen0 Standard Mono shard, both seats
#   gen0-mono/train      the rest of those shards: a large training set whose opponents exclude the whole Pauper pool
#   gen0-pauper/test     the first 100 games of each gen0 Pauper non-mirror shard, both seats (unseen by gen0-mono)
#   kinds/<kind>/train|test  gen1 by kind (reg, selfplay, pimc, other), every 5th game held out; kinds/src/<kind>: all
#   branch-A|B|C|D/train     all of gen0 + gen1 reg (A), selfplay (B), both (C), everything (D)
set -e
R=/workspace/repo/experiment_data/Pauper-Faeries D=/workspace/data
for seat in PlayerA PlayerB; do
  mkdir -p $D/gen0/train/$seat
  ln -sf $R/ver4/training/gen0/$seat/*Pauper-*_vs_Pauper-*.hdf5 $D/gen0/train/$seat/
done
mkdir -p $D/gen0 && ln -sfn $R/ver4/testing/gen0 $D/gen0/test
cd /workspace/repo/src/magezero/util
# gen1/train|test were split before the Oct 29 reorganization of the gen1 folders and are kept as built
mkdir -p $D/both/train && cp -rsf $D/gen0/train/. $D/both/train/gen0 && cp -rsf $D/gen1/train/. $D/both/train/gen1
# the seat's own shards only: gen0 PlayerB also holds a copy of PlayerA's MonoW shard
for seat in "PlayerA Pauper-Faeries_vs_Standard-Mono" "PlayerB Standard-Mono"; do
  set -- $seat
  mkdir -p $D/gen0-nomirror/train/$1 $D/gen0-mono/test/$1
  ln -sf $(ls $D/gen0/train/$1/*.hdf5 | grep -v Pauper-Faeries_vs_Pauper-Faeries) $D/gen0-nomirror/train/$1/
  mkdir -p $D/gen0-mono/train/$1 $D/gen0-pauper/test/$1
  for f in $R/ver4/training/gen0/$1/$2*.hdf5; do
    [ -f $D/gen0-mono/train/$1/$(basename $f) ] && continue
    python3 cut_games.py $f $D/gen0-mono/test/$1/$(basename $f) 0 100
    python3 cut_games.py $f $D/gen0-mono/train/$1/$(basename $f) 100 1000000
  done
  for f in $(ls $D/gen0-nomirror/train/$1/*.hdf5); do
    [ -f $D/gen0-pauper/test/$1/$(basename $f) ] || python3 cut_games.py $f $D/gen0-pauper/test/$1/$(basename $f) 0 100
  done
done
# gen1 by kind (src/<kind>/<seat>, links to the repo's shards), each split by game: every 5th held out
#   selfplay  `*_selfplay*`: v4 model on both seats
#   pimc      `*_pimc*`: hidden information
#   other     `*_policy*` (priors on) and ver5 (v5 model)
#   reg       the rest of ver4 gen1: v4 model against the heuristic search
K=$D/kinds
for seat in PlayerA PlayerB; do
  mkdir -p $K/src/{reg,selfplay,pimc,other}/$seat
  for f in $R/ver4/training/gen1/$seat/*.hdf5; do
    case $(basename $f) in *_selfplay*) k=selfplay;; *_pimc*) k=pimc;; *_policy*) k=other;; *) k=reg;; esac
    ln -sf $f $K/src/$k/$seat/
  done
  for f in $R/ver5/training/gen1/$seat/*.hdf5; do ln -sf $f $K/src/other/$seat/ver5-$(basename $f); done
done
for k in reg selfplay pimc other; do python3 split_games.py $K/src/$k $K/$k/train $K/$k/test --every 5; done
# gen0: every ver4 gen0 shard, both seats, without PlayerB's copy of PlayerA's MonoW shard
for seat in PlayerA PlayerB; do
  mkdir -p $D/gen0-all/$seat
  ln -sf $(ls $R/ver4/training/gen0/$seat/*.hdf5 | grep -v "PlayerB/Pauper-Faeries_vs_Standard-MonoW") $D/gen0-all/$seat/
done
# branches: gen0 + A reg, B selfplay, C reg + selfplay, D everything
for b in A B C D; do rm -rf $D/branch-$b && mkdir -p $D/branch-$b/train && cp -rs $D/gen0-all/. $D/branch-$b/train/gen0; done
for b in A C D; do cp -rs $K/reg/train/. $D/branch-$b/train/reg; done
for b in B C D; do cp -rs $K/selfplay/train/. $D/branch-$b/train/selfplay; done
cp -rs $K/pimc/train/. $D/branch-D/train/pimc && cp -rs $K/other/train/. $D/branch-D/train/other
du -shL $D/*/*
df -h /workspace
