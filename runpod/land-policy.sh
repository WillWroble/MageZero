# util/land_policy.py for $MODELS ("label=path ...", paths under /workspace/runs or absolute; label=P1 is
# experiment_models/Pauper/P1) on P1's Burn mirror (300- and 1000-sim A shards) and the held-out gen0 Faeries mirror
# (PlayerA, PlayerB). Output: $OUT/land-policy.txt.
apt-get update -qq && apt-get install -y -qq git-lfs python3 python3-pip >/dev/null
pip3 install -q h5py numpy torch
D=experiment_data/Pauper/ver1; V=/workspace/runs/experiment_data/Pauper-Faeries/ver4/testing/gen0; P1=experiment_models/Pauper/P1/model.pt.gz
git lfs install --local >/dev/null && git lfs pull --include="$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5,$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5,$P1"
A=""; for m in $MODELS; do l=${m%%=*}; p=${m#*=}; [ "$l" = P1 ] && { A="$A P1=$PWD/$P1"; continue; }
  case $p in /*) ;; *) p=/workspace/runs/$p;; esac; [ -f $p ] && A="$A $l=$p" || echo "missing $p"; done
cd src/magezero && PYTHONPATH=. python3 util/land_policy.py $A -- burn300=../../$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5 \
  burn1000=../../$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5 faeA=$V/PlayerA/Pauper-Faeries_vs_Pauper-Faeries.hdf5 \
  faeB=$V/PlayerB/Pauper-Faeries_vs_Pauper-Faeries.hdf5 > $OUT/land-policy.txt
