# Game generation in a Kaggle GPU kernel (run by kaggle/generate.py; runpod/generate.sh is the RunPod twin):
# serves the model from the attached dataset on port 50052 and plays $CONFIG with XMage at $MAGE_COMMIT, built here
# or unpacked from an earlier kernel's xmage.tar.gz. Outputs in /kaggle/working: the config's shards, logs, results.
set -ex
W=/kaggle/working
nproc; lscpu > $W/cpu.txt; free -g > $W/mem.txt; nvidia-smi > $W/gpu.txt || true

# Java 21 (Temurin), as in runpod/java.sh
T=/tmp/tools && mkdir -p $T/jdk
curl -sSL --retry 5 --retry-all-errors -o $T/jdk.tar.gz "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
tar xzf $T/jdk.tar.gz -C $T/jdk --strip-components=1
export JAVA_HOME=$T/jdk PATH=$T/jdk/bin:$PATH

# XMage, as in runpod/build-xmage.sh; the build is saved as xmage.tar.gz for later kernels (--xmage)
X=/tmp/xmage && mkdir -p $X
PREBUILT=$(find /kaggle/input -name xmage.tar.gz | head -1)
if [ -n "$PREBUILT" ]; then
  tar xzf $PREBUILT -C $X
else
  mkdir -p $T/maven && curl -sSL --retry 5 --retry-all-errors -o $T/maven.tar.gz \
    https://repo1.maven.org/maven2/org/apache/maven/apache-maven/3.9.11/apache-maven-3.9.11-bin.tar.gz
  tar xzf $T/maven.tar.gz -C $T/maven --strip-components=1
  git clone -q --filter=blob:none https://github.com/WillWroble/mage /tmp/mage && git -C /tmp/mage checkout -q $MAGE_COMMIT
  (cd /tmp/mage && MAVEN_OPTS=-Xmx12g $T/maven/bin/mvn -B -q -T 1C install -DskipTests -Dmaven.repo.local=/tmp/m2 -pl Mage.MageZero -am)
  unzip -q -o /tmp/mage/Mage.MageZero/target/mage-magezero.zip -d $X
  cp -r /tmp/mage/Mage.MageZero/decks /tmp/mage/Mage.MageZero/config $X/
  tar czf $W/xmage.tar.gz -C $X .
fi

# the model server
pip install -q flask waitress msgpack
M=/tmp/model && mkdir -p $M
F=$(find /kaggle/input -name "model.pt*" | head -1)
case $F in *.gz) cp $F $M/model.pt.gz;; *) gzip -c $F > $M/model.pt.gz;; esac  # Kaggle stores .gz files unpacked
python -u $SRC/server.py --model $M --port 50052 > >(grep --line-buffered -v "^\[REQ" > $W/server.log) 2>&1 &
SERVER=$!
trap "kill $SERVER 2>/dev/null || true" EXIT  # the server's log stream below ends with it, so the kernel can finish
(set +x; tail --pid=$SERVER -f $W/server.log | grep --line-buffered -v "^\[BATCH" | sed -u "s/^/[server] /") &
# wait for the server; stop if it dies or isn't up within 5 minutes
(set +x; for i in $(seq 150); do
  curl -sf localhost:50052/healthz && exit 0
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  sleep 2
done; echo "server not up after 5 minutes"; exit 1)

# the games: 2 threads per vCPU, the JVM heap is the RAM left after 8 GB for the server and the system
sed "s|^  threads: .*|  threads: $((2 * $(nproc)))|" $CONFIG > $W/config.yml
grep output_file $W/config.yml | awk '{print $2}' | xargs -n1 dirname | xargs mkdir -p
HEAP=$(( $(free -g | awk '/^Mem:/ {print $2}') - 8 ))
# every 5 minutes: games finished and the current win rate (the generator's "Game #N completed" / "Current WR" lines)
(set +x; while sleep 300; do
  echo "[progress] $(date -u +%H:%M) games $(grep -c 'completed successfully' /tmp/game.log) $(grep 'Current WR' /tmp/game.log | tail -1)"
done) &
PROGRESS=$!
(cd $X && timeout $GEN_SECONDS java -Xms2g -Xmx${HEAP}g -XX:+UseParallelGC --add-opens=java.base/java.lang=ALL-UNNAMED \
  -Dlog4j.configuration=file:config/log4j.properties -jar lib/mage-magezero-1.4.58.jar $W/config.yml > /tmp/game.log 2>&1) \
  || echo "java exited with $?"
kill $PROGRESS
tail -n 2000 /tmp/game.log > $W/java-tail.log
grep -E "Successful:|Failed:|win rate|WR with" /tmp/game.log > $W/result.txt || true
grep -c "ERROR" /tmp/game.log > $W/error-count.txt || true
cp $X/WinRates.txt $W/ 2>/dev/null || true
