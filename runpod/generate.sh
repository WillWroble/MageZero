# Self-play data generation on a GPU pod: serves $MODEL (a model directory under the repo) for both seats on port
# 50052 and runs the XMage distribution at /workspace/xmage/$MAGE_COMMIT (built first if missing) with $CONFIG (a file in the repo), whose
# output_file paths should point into $OUT. Run by launch.py --script.
set -ex
apt-get install -y -qq curl unzip git-lfs >/dev/null
bash runpod/build-xmage.sh  # once per mage commit
source runpod/java.sh
pip install -q h5py flask waitress msgpack pyyaml
lscpu > $OUT/cpu.txt; nvidia-smi > $OUT/gpu.txt
# the model, from the volume's copy of the repo
(cd /workspace/repo && git lfs pull --include="$MODEL/model.pt.gz")
mkdir -p models/Pauper-Faeries/ver4 && cp /workspace/repo/$MODEL/model.pt.gz models/Pauper-Faeries/ver4/
python -u src/magezero/server.py --deck Pauper-Faeries --version 4 --port 50052 2>&1 | grep -v "^\[REQ" > $OUT/server.log &
until curl -sf localhost:50052/healthz; do sleep 2; done
X=/workspace/xmage/$MAGE_COMMIT
cp $CONFIG $OUT/config.yml
# the game log is large: it stays on the container disk, with its tail copied to the volume
(while sleep 60; do tail -n 500 /tmp/java.log > $OUT/java-tail.log; done) &
cd $X && java -Xms2g -Xmx48g -XX:+UseParallelGC --add-opens=java.base/java.lang=ALL-UNNAMED \
  -jar lib/mage-magezero-1.4.58.jar $OUT/config.yml > /tmp/java.log 2>&1
tail -n 2000 /tmp/java.log > $OUT/java-tail.log
grep "WR with" $X/WinRates.txt | tail -3 > $OUT/winrate.txt || true
