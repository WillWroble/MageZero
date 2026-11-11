# Runs util/land_value.py on ver15's and gen0's Faeries shards against each Pauper deck, MonoB and MonoR, straight
# from the volume. Output: $OUT/ver15.jsonl, $OUT/gen0.jsonl.
pip install -q h5py numpy torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1 || pip install -q h5py numpy torch
cd src/magezero
R=/workspace/runs; G0=/workspace/repo/experiment_data/Pauper-Faeries/ver4/training/gen0/PlayerA
V=""; Z=""
for o in Affinity Burn CawGates Elves Faeries Rally Spy Wildfire; do
  V="$V $o=$R/gen5-offline/PlayerA/Pauper-Faeries_vs_Pauper-$o.hdf5"; Z="$Z $o=$G0/Pauper-Faeries_vs_Pauper-$o.hdf5"; done
V="$V MonoB=$R/norm-F2-gnn/PlayerA/Pauper-Faeries_vs_Standard-MonoB.hdf5 MonoR-ver13=$R/gen3data-offline/PlayerA/Pauper-Faeries_vs_Standard-MonoR.hdf5"
Z="$Z MonoB=$G0/Pauper-Faeries_vs_Standard-MonoB.hdf5 MonoR=$G0/Pauper-Faeries_vs_Standard-MonoR.hdf5"
PYTHONPATH=. python util/land_value.py $OUT/ver15.jsonl $V &
PYTHONPATH=. python util/land_value.py $OUT/gen0.jsonl $Z &
wait
