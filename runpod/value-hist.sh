# Value label and root value histograms of every shard in the volume's repo checkout: $OUT/value-hist.json
set -ex
apt-get install -y -qq python3-pip >/dev/null && pip install -q h5py numpy
python3 src/magezero/util/value_hist.py /workspace/repo/experiment_data $OUT/value-hist.json
