# Value predictions of each model in $MODELS on each shard directory in $DATA: $OUT/<dir>.npz (util/value_preds.py)
set -ex
pip install -q h5py
# the repo's own models, for $MODELS under /tmp/code/experiment_models
apt-get install -y -qq git-lfs >/dev/null && git lfs install --local && git lfs pull --include="experiment_models/**/model.pt.gz"
cd src/magezero
for d in $DATA; do PYTHONPATH=. python util/value_preds.py $OUT/$(echo ${d#/workspace/data/} | tr / -).npz $d $MODELS; done
