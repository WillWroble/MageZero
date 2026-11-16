# Checkpoints re-scored on common labels: $CKPTS = "label=run:ckpt ..." (checkpoint names in /workspace/runs/<run>/gnn).
# Each is copied to $OUT/<label>/model.pt.gz with its run's split.txt and tested (test.py, held-out sets) against value
# labels rebuilt with TD 0.92 and against the game result; then land-inject.sh and land-policy.sh run on all of them.
# Output: $OUT/<label>/test.log, $OUT/inject.txt, $OUT/inject-faeries.txt, $OUT/land-policy.txt.
pip3 install -q h5py numpy
M=""
for a in $CKPTS; do l=${a%%=*}; r=${a#*=}; run=${r%%:*}
  mkdir -p $OUT/$l && cp /workspace/runs/$run/gnn/${r#*:} $OUT/$l/model.pt.gz && cp /workspace/runs/$run/gnn/split.txt $OUT/$l/
  for td in 0.92 1; do
    (cd /workspace/repo && PYTHONPATH=/tmp/code/src/magezero python3 /tmp/code/src/magezero/test.py --model $OUT/$l --td $td > /dev/null)
  done
  M="$M $l=${OUT#/workspace/runs/}/$l/model.pt.gz"
done
MODELS="$M" bash runpod/land-inject.sh
MODELS="$M" bash runpod/land-policy.sh
