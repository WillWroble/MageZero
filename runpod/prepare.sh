# Experiment data sets under /workspace/data, from the repo's shards (run by files.sh):
#   gen0/train  ver4 gen0, Pauper pool (8 matchups incl. the mirror), both seats (the set Dama compared on)
#   gen0/test   ver4 gen0 mirror test set
#   gen1/train, gen1/test  every gen1 shard (ver4, ver5) except the hidden-information _pimc ones, split by game:
#               the last 20% of each shard's games are held out
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
du -shL $D/*/*
