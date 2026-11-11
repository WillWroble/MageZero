# util/td_adaptive.py on ver15's Faeries shards vs each Pauper deck and MonoB (ver13 for MonoR), from the volume.
apt-get update -qq && apt-get install -y -qq python3 python3-pip >/dev/null
pip3 install -q h5py numpy
R=/workspace/runs; A=""
for o in Affinity Burn CawGates Elves Faeries Rally Spy Wildfire; do A="$A $o=$R/gen5-offline/PlayerA/Pauper-Faeries_vs_Pauper-$o.hdf5"; done
A="$A MonoB=$R/norm-F2-gnn/PlayerA/Pauper-Faeries_vs_Standard-MonoB.hdf5 MonoR-ver13=$R/gen3data-offline/PlayerA/Pauper-Faeries_vs_Standard-MonoR.hdf5"
python3 src/magezero/util/td_adaptive.py $OUT/td.json $A
