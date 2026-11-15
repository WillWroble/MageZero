# util/land_inject.py for $MODELS ("label=path ...", paths under /workspace/runs or absolute) on P1's Burn mirror
# (300- and 1000-sim A shards; Mountain) and the held-out gen0 Faeries mirror on the volume (PlayerA, PlayerB; Island).
# Output: $OUT/inject.txt (Burn), $OUT/inject-faeries.txt.
apt-get update -qq && apt-get install -y -qq git-lfs python3 python3-pip >/dev/null
pip3 install -q h5py numpy torch
D=experiment_data/Pauper/ver1; V=/workspace/runs/experiment_data/Pauper-Faeries/ver4/testing/gen0
git lfs install --local >/dev/null && git lfs pull --include="$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5,$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5"
A=""; for m in $MODELS; do l=${m%%=*}; p=${m#*=}; case $p in /*) ;; *) p=/workspace/runs/$p;; esac
  [ -f $p ] && A="$A $l=$p" || echo "missing $p"; done
cd src/magezero && PYTHONPATH=. python3 util/land_inject.py $A -- ../../$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5 ../../$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5 > $OUT/inject.txt
LAND=Island PYTHONPATH=. python3 util/land_inject.py $A -- A=$V/PlayerA/Pauper-Faeries_vs_Pauper-Faeries.hdf5 B=$V/PlayerB/Pauper-Faeries_vs_Pauper-Faeries.hdf5 > $OUT/inject-faeries.txt
