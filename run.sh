#!/usr/bin/env bash
# Builds the prototype and runs one of the benchmarks.
#
#   ./run.sh srk   validate 2000                  # ln phi and Z vs NeqSim
#   ./run.sh srk   bench 1000 10000 100000 1000000
#   ./run.sh flash validate 20000                 # flash vs NeqSim TPflash
#   ./run.sh flash bench 100000 1000000
#
# BACKEND=cuda|opencl|metal picks the TornadoVM backend (default cuda), DEVICE its device index.
# Extra JVM options go in JAVA_OPTS, e.g. JAVA_OPTS="-DmaxIt=50 -Dreps=9".
set -euo pipefail
cd "$(dirname "$0")"
BACKEND=${BACKEND:-cuda}
DEVICE=${DEVICE:-0}
case "${1:-}" in
  srk) MAIN=SrkBatchBench ;;
  flash) MAIN=SrkFlashBench ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
shift
mvn -q -Dtornadovm.backend="$BACKEND" compile dependency:build-classpath -Dmdep.outputFile=cp.txt
exec java -Xmx24g --enable-native-access=ALL-UNNAMED \
  -Dtornado.enabled.backends="$BACKEND" -Dtornado.backend.device="0:$DEVICE" \
  ${JAVA_OPTS:-} -cp "target/classes:$(cat cp.txt)" \
  "io.github.mikepapadim.neqsim.tornadovm.$MAIN" "$@"
