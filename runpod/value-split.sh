# Value predictions (util/value_preds.py) of the model directories $MODELS (under /workspace/runs) on each val set in
# $SETS of $SPLIT (a split.txt under /workspace/runs; its regexes match the files pod's checkout): $OUT/<set>.npz
pip3 install -q h5py numpy
M=""; for m in $MODELS; do M="$M /workspace/runs/$m"; done
cd /workspace/repo && for s in $SETS; do
  PYTHONPATH=/tmp/code/src/magezero python3 /tmp/code/src/magezero/util/value_preds.py $OUT/$s.npz /workspace/runs/$SPLIT:$s $M
done
