# Data generation on a GPU pod: serves $MODEL (a model directory under the repo) on port 50052, for each seat that
# isn't offline, and runs (a local copy of) the XMage distribution at /workspace/xmage/$MAGE_COMMIT (built first if missing) with each config
# in $CONFIG (space-separated files in the repo), one after another. Each config's output_file paths say where its
# shards go; its game log tail and sims/s lines land in $OUT. Run by launch.py --script.
set -ex
apt-get install -y -qq curl unzip git-lfs >/dev/null
bash runpod/build-xmage.sh  # once per mage commit
source runpod/java.sh
pip install -q h5py flask waitress msgpack pyyaml
lscpu > $OUT/cpu.txt; nvidia-smi > $OUT/gpu.txt
# the model, from this pod's own clone (at the launched commit)
git lfs install --local >/dev/null && git lfs pull --include="$MODEL/model.pt.gz"
python -u src/magezero/server.py --model $MODEL --port 50052 2>&1 | grep -v "^\[REQ" > $OUT/server.log &
until curl -sf localhost:50052/healthz; do sleep 2; done
# $OPP_MODEL (optional): a second model on port 50053, for configs whose server.opponent_port points player_b there
if [ -n "$OPP_MODEL" ]; then
  git lfs pull --include="$OPP_MODEL/model.pt.gz"
  python -u src/magezero/server.py --model $OPP_MODEL --port 50053 2>&1 | grep -v "^\[REQ" > $OUT/server-opp.log &
  until curl -sf localhost:50053/healthz; do sleep 2; done
fi
# a local copy per pod, without the card database: XMage builds its own (db/) on startup. A shared or copied
# database breaks when another pod is using it (failed connections, decks loading with cards missing)
mkdir -p /tmp/xmage && cp -r /workspace/xmage/$MAGE_COMMIT/{lib,decks,config} /tmp/xmage/ && X=/tmp/xmage
for C in $CONFIG; do
  N=$(basename $C .yml)
  cp $C $OUT/$N.yml
  grep output_file $C | awk '{print $2}' | xargs -n1 dirname | xargs mkdir -p
  # the game log is large: it stays on the container disk, with its tail and the sims/s lines copied to the volume
  (while sleep 60; do tail -n 500 /tmp/$N.log > $OUT/$N-java-tail.log; grep "Total: simulated" /tmp/$N.log > $OUT/$N-sims.log || true; done) &
  LOOP=$!
  (cd $X && java -Xms2g -Xmx48g -XX:+UseParallelGC --add-opens=java.base/java.lang=ALL-UNNAMED \
    -Dlog4j.configuration=file:config/log4j.properties -jar lib/mage-magezero-1.4.58.jar $OUT/$N.yml > /tmp/$N.log 2>&1)
  kill $LOOP || true  # (set -e)
  tail -n 2000 /tmp/$N.log > $OUT/$N-java-tail.log
  grep "Total: simulated" /tmp/$N.log > $OUT/$N-sims.log || true
  grep -E "Successful:|Failed:|win rate" /tmp/$N.log > $OUT/$N-result.txt || true
done
