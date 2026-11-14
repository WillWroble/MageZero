# Checkpoint averages: $AVG = "label=run:ckpt,ckpt,... ..." (checkpoint names in /workspace/runs/<run>/gnn, or paths
# under /workspace/runs). Each average is written to $OUT/<label>/model.pt.gz with its run's split.txt and tested there
# (test.py, held-out sets), then land-inject.sh and ablate.sh run on all of them. Output: $OUT/<label>/test.log,
# $OUT/inject.txt, $OUT/ablate.txt.
pip3 install -q h5py numpy
M=""
for a in $AVG; do l=${a%%=*}; r=${a#*=}; run=${r%%:*}; C=""
  for c in $(echo ${r#*:} | tr , ' '); do case $c in */*) C="$C /workspace/runs/$c";; *) C="$C /workspace/runs/$run/gnn/$c";; esac; done
  mkdir -p $OUT/$l && cp /workspace/runs/$run/gnn/split.txt $OUT/$l/
  (cd src/magezero && PYTHONPATH=. python3 util/average.py $OUT/$l/model.pt.gz $C)
  (cd /workspace/repo && PYTHONPATH=/tmp/code/src/magezero python3 /tmp/code/src/magezero/test.py --model $OUT/$l > /dev/null)
  M="$M $l=${OUT#/workspace/runs/}/$l/model.pt.gz"
done
MODELS="$M" bash runpod/land-inject.sh
MODELS="$M" N=2000 bash runpod/ablate.sh
