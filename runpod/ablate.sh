# util/ablate.py on $N states of the Burn mirror (P1's 300- and 1000-sim A shards) for $MODELS ("label=run ...":
# /workspace/runs/<run>/gnn/model.pt.gz; label=P1 is experiment_models/Pauper/P1). Output: $OUT/ablate.txt.
apt-get update -qq && apt-get install -y -qq git-lfs python3 python3-pip >/dev/null
pip3 install -q h5py numpy torch
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/WillWroble/mage /tmp/mage \
  && git -C /tmp/mage sparse-checkout set Mage.MageZero/decks Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai/encoder
D=experiment_data/Pauper/ver1
git lfs install --local >/dev/null && git lfs pull --include="$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5,$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5,experiment_models/Pauper/P1/model.pt.gz"
A=""; for m in $MODELS; do l=${m%%=*}; r=${m#*=}
  [ "$l" = P1 ] && A="$A P1=$PWD/experiment_models/Pauper/P1/model.pt.gz" || A="$A $l=/workspace/runs/$r/gnn/model.pt.gz"; done
cd src/magezero && MAGE_SRC=/tmp/mage PYTHONPATH=. python3 util/ablate.py $OUT/ablate.txt ${N:-2000} $A -- ../../$D/Pauper-Burn_vs_Pauper-Burn300A.hdf5 ../../$D/Pauper-Burn_vs_Pauper-Burn1000A.hdf5
