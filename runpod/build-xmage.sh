# Builds the MageZero XMage distribution at $MAGE_COMMIT (WillWroble/mage) into /workspace/xmage/$MAGE_COMMIT:
# lib/ (the jars), decks/ and config/ from Mage.MageZero. Run on a CPU pod by launch.py --script.
set -ex
apt-get install -y -qq curl unzip >/dev/null
source runpod/java.sh
X=/workspace/xmage/$MAGE_COMMIT
[ -f $X/lib/mage-magezero-1.4.58.jar ] && exit 0
git clone -q --filter=blob:none https://github.com/WillWroble/mage /tmp/mage && cd /tmp/mage && git checkout -q $MAGE_COMMIT
MAVEN_OPTS=-Xmx24g mvn -B -q -T 1C install -DskipTests -Dmaven.repo.local=/workspace/m2 -pl Mage.MageZero -am
mkdir -p $X && unzip -q -o Mage.MageZero/target/mage-magezero.zip -d $X
cp -r Mage.MageZero/decks Mage.MageZero/config $X/
ls -la $X $X/lib | head -20
