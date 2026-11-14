# The Faeries mirror games of v4-dd-players-noname (all-node) and v4-dd-players-legal-noname (legal-only), Nov 13
# (S=-h5: their holdout retrains' games, Nov 14: 200 vs offline each, one head-to-head with all-node as seat A):
# util/land_pass.py on each model's own decisions, and both models' value MSE on every shard (util/value_preds.py).
# Output: $OUT/land.txt, $OUT/value.txt.
apt-get update -qq && apt-get install -y -qq python3 python3-pip >/dev/null
pip3 install -q h5py numpy torch
cd src/magezero
R=/workspace/runs; f=Pauper-Faeries_vs_Pauper-Faeries.hdf5
if [ "$S" = -h5 ]; then
  N="norm-all-nn-h5 norm-legal-nn-h5 h2h-allA-legalB-h5"
  A="all-vs-offline=$R/norm-all-nn-h5/PlayerA/$f legal-vs-offline=$R/norm-legal-nn-h5/PlayerA/$f"
  A="$A all-h2h-seatA=$R/h2h-allA-legalB-h5/PlayerA/$f legal-h2h-seatB=$R/h2h-allA-legalB-h5/PlayerB/$f"
else
  N="norm-all-nn-1 norm-all-nn-2 norm-legal-nn-1 norm-legal-nn-2 h2h-allA-legalB h2h-legalA-allB"
  A=""; for i in 1 2; do A="$A all-vs-offline-$i=$R/norm-all-nn-$i/PlayerA/$f legal-vs-offline-$i=$R/norm-legal-nn-$i/PlayerA/$f"; done
  A="$A all-h2h-seatA=$R/h2h-allA-legalB/PlayerA/$f legal-h2h-seatB=$R/h2h-allA-legalB/PlayerB/$f"
  A="$A legal-h2h-seatA=$R/h2h-legalA-allB/PlayerA/$f all-h2h-seatB=$R/h2h-legalA-allB/PlayerB/$f"
  A="$A gen0=/workspace/repo/experiment_data/Pauper-Faeries/ver4/training/gen0/PlayerA/$f ver15=$R/gen5-offline/PlayerA/$f ddp=$R/norm-ddplayers/PlayerA/$f"
fi
PYTHONPATH=. python3 util/land_pass.py $A > $OUT/land.txt
for d in $N; do for s in PlayerA PlayerB; do
  echo "== $d/$s" >> $OUT/value.txt
  PYTHONPATH=. python3 util/value_preds.py /tmp/v.npz $R/$d/$s $R/v4-dd-players-noname$S/gnn $R/v4-dd-players-legal-noname$S/gnn 2>&1 | grep "MSE vs label" >> $OUT/value.txt
done; done
