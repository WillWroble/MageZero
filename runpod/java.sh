# Java 21 (Temurin) and Maven 3.9 on the volume, installed once; sourced by the XMage scripts.
T=/workspace/tools
if [ ! -x $T/jdk/bin/java ]; then
  mkdir -p $T/jdk && curl -sSL "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse" \
    | tar xz -C $T/jdk --strip-components=1
fi
if [ ! -x $T/maven/bin/mvn ]; then
  mkdir -p $T/maven && curl -sSL https://archive.apache.org/dist/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz \
    | tar xz -C $T/maven --strip-components=1
fi
export JAVA_HOME=$T/jdk PATH=$T/jdk/bin:$T/maven/bin:$PATH
