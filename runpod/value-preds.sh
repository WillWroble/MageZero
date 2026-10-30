# Value predictions of each model in $MODELS on each shard directory in $DATA: $OUT/<dir>.npz (util/value_preds.py)
set -ex
pip install -q h5py
cd src/magezero
for d in $DATA; do PYTHONPATH=. python util/value_preds.py $OUT/$(echo ${d#/workspace/data/} | tr / -).npz $d $MODELS; done
