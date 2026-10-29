# Experiment data sets under /workspace/data, from the repo's shards (run by files.sh):
#   gen0/train  ver4 gen0, Pauper pool (8 matchups incl. the mirror), both seats (the set Dama compared on)
#   gen0/test   ver4 gen0 mirror test set
#   gen1/train, gen1/test  every gen1 shard (ver4, ver5) except the hidden-information _pimc ones, split by game:
#               the last 20% of each shard's games are held out
#   both/train  gen0/train and gen1/train
#   gen0-nomirror/train  gen0/train without the mirror shards: the mirror is then an unseen opponent
#   gen0-mono/test       the first 100 games of each gen0 Standard Mono shard, both seats
#   gen0-mono/train      the rest of those shards: a large training set whose opponents exclude the whole Pauper pool
#   gen0-pauper/test     the first 100 games of each gen0 Pauper non-mirror shard, both seats (unseen by gen0-mono)
set -e
R=/workspace/repo/experiment_data/Pauper-Faeries D=/workspace/data
for seat in PlayerA PlayerB; do
  mkdir -p $D/gen0/train/$seat
  ln -sf $R/ver4/training/gen0/$seat/*Pauper-*_vs_Pauper-*.hdf5 $D/gen0/train/$seat/
done
mkdir -p $D/gen0 && ln -sfn $R/ver4/testing/gen0 $D/gen0/test
cd /workspace/repo/src/magezero/util
for v in ver4 ver5; do
  python3 split_games.py $R/$v/training/gen1 $D/gen1/train/$v $D/gen1/test/$v 0.2 _pimc
done
mkdir -p $D/both/train && cp -rsf $D/gen0/train/. $D/both/train/gen0 && cp -rsf $D/gen1/train/. $D/both/train/gen1
# the seat's own shards only: gen0 PlayerB also holds a copy of PlayerA's MonoW shard
rm -rf $D/gen0-mono
for seat in "PlayerA Pauper-Faeries_vs_Standard-Mono" "PlayerB Standard-Mono"; do
  set -- $seat
  mkdir -p $D/gen0-nomirror/train/$1 $D/gen0-mono/test/$1
  ln -sf $(ls $D/gen0/train/$1/*.hdf5 | grep -v Pauper-Faeries_vs_Pauper-Faeries) $D/gen0-nomirror/train/$1/
  mkdir -p $D/gen0-mono/train/$1 $D/gen0-pauper/test/$1
  for f in $R/ver4/training/gen0/$1/$2*.hdf5; do
    python3 cut_games.py $f $D/gen0-mono/test/$1/$(basename $f) 0 100
    python3 cut_games.py $f $D/gen0-mono/train/$1/$(basename $f) 100 1000000
  done
  for f in $(ls $D/gen0-nomirror/train/$1/*.hdf5); do
    python3 cut_games.py $f $D/gen0-pauper/test/$1/$(basename $f) 0 100
  done
done
du -shL $D/*/*
