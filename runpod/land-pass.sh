# util/land_pass.py on Faeries vs offline Pauper shards from the volume: gen0 (heuristic), gen1 (v4; only Affinity, Burn,
# CawGates, Elves are in the repo), gen5 (ver15) and $RUN (a norm-* run's directory). Output: $OUT/land.txt.
apt-get update -qq && apt-get install -y -qq python3 python3-pip >/dev/null
pip3 install -q h5py numpy && pip3 install -q torch --index-url https://download.pytorch.org/whl/cpu
cd src/magezero
R=/workspace/runs; T=/workspace/repo/experiment_data/Pauper-Faeries/ver4/training
for o in Affinity Burn CawGates Elves Faeries Rally Spy Wildfire; do
  f=Pauper-Faeries_vs_Pauper-$o.hdf5
  A="$A gen0-$o=$T/gen0/PlayerA/$f"; [ -s $T/gen1/PlayerA/$f ] && A="$A v4-$o=$T/gen1/PlayerA/$f"
  A="$A ver15-$o=$R/gen5-offline/PlayerA/$f $(basename $RUN)-$o=$RUN/PlayerA/$f"
done
PYTHONPATH=. python3 util/land_pass.py $A > $OUT/land.txt
