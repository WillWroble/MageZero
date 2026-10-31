# util/behaviour.py and results/pimc_play.py on each shard in $SHARDS (paths on the volume), in parallel:
# $OUT/beh-<shard>.txt and $OUT/play-<shard>.txt
set -ex
pip install -q h5py
cd src/magezero
printf '%s\n' $SHARDS | xargs -P 16 -I{} bash -c 'n=$(echo {} | tr / _); PYTHONPATH=. python util/behaviour.py {} > $OUT/beh-$n.txt 2>&1; PYTHONPATH=. python ../../results/pimc_play.py {} > $OUT/play-$n.txt 2>&1'
