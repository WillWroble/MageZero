# Self-play data generation on a GPU pod: serves $MODEL (a model directory under the repo) for both seats on port
# 50052 and runs the XMage distribution at /workspace/xmage/$MAGE_COMMIT (built first if missing) with each config
# in $CONFIG (space-separated files in the repo), one after another. Each config's output_file paths say where its
# shards go; its game log tail and sims/s lines land in $OUT. Run by launch.py --script.
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
for C in $CONFIG; do
  N=$(basename $C .yml)
  cp $C $OUT/$N.yml
  grep output_file $C | awk '{print $2}' | xargs -n1 dirname | xargs mkdir -p
  # the game log is large: it stays on the container disk, with its tail and the sims/s lines copied to the volume
  (while sleep 60; do tail -n 500 /tmp/$N.log > $OUT/$N-java-tail.log; grep "Total: simulated" /tmp/$N.log > $OUT/$N-sims.log; done) &
  LOOP=$!
  (cd $X && java -Xms2g -Xmx48g -XX:+UseParallelGC --add-opens=java.base/java.lang=ALL-UNNAMED \
    -Dlog4j.configuration=file:config/log4j.properties -jar lib/mage-magezero-1.4.58.jar $OUT/$N.yml > /tmp/$N.log 2>&1)
  kill $LOOP
  tail -n 2000 /tmp/$N.log > $OUT/$N-java-tail.log
  grep "Total: simulated" /tmp/$N.log > $OUT/$N-sims.log || true
  grep -E "Successful:|Failed:|win rate" /tmp/$N.log > $OUT/$N-result.txt || true
done
