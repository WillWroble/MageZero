# Value label and root value histograms of every shard in the volume's repo checkout: $OUT/value-hist.json
# ($TD: discounts to recompute the labels with, e.g. TD="0.97 0.99")
set -ex
apt-get install -y -qq python3-pip >/dev/null && pip install -q h5py numpy
python3 src/magezero/util/value_hist.py /workspace/repo/experiment_data $OUT/value-hist.json $TD
