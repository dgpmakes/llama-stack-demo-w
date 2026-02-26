#!/usr/bin/env bash
# Verifies that monitoring and telemetry resources exist and are ready in the
# cluster (namespace, Tempo, OpenTelemetry collector, Prometheus, instrumentation).
# Run after ./scripts/setup-monitoring-for-helm.sh and once the operators have reconciled.
#
# Usage: ./scripts/check-monitoring-telemetry.sh [--lenient] [--skip-prometheus] [--skip-instrumentation]
# Env:   MONITORING_NAMESPACE  (default: redhat-ods-monitoring)
# Exit:  0 if all checks pass, 1 otherwise

set -euo pipefail

MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-redhat-ods-monitoring}"
SKIP_PROMETHEUS=false
SKIP_INSTRUMENTATION=false
LENIENT=false
for arg in "$@"; do
  case "$arg" in
    --lenient)              LENIENT=true ;;
    --skip-prometheus)      SKIP_PROMETHEUS=true ;;
    --skip-instrumentation) SKIP_INSTRUMENTATION=true ;;
    -h|--help)
      echo "Usage: $0 [--lenient] [--skip-prometheus] [--skip-instrumentation]" >&2
      echo "  --lenient              Only check CRs and services exist; skip pod-running and Instrumentation (use right after setup)." >&2
      echo "  --skip-prometheus      Do not require Prometheus service (data-science-monitoringstack-prometheus)." >&2
      echo "  --skip-instrumentation Do not require OpenTelemetry Instrumentation (data-science-instrumentation)." >&2
      echo "Env:  MONITORING_NAMESPACE (default: redhat-ods-monitoring)" >&2
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

FAILED=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "  OK   $name"
  else
    echo "  FAIL $name"
    FAILED=1
  fi
}

# Try multiple label selectors; succeed if any yields >= min Running pods.
# Fallback: count Running pods whose name matches name_pattern (regex).
# Usage: check_pods_running "label" "ns" min "name_pattern" "selector1" ["selector2" ...]
check_pods_running() {
  local name="$1"
  local ns="$2"
  local min="${3:-1}"
  local name_pattern="${4:-.}"
  shift 4
  local count=0
  local selector
  # oc get pods --no-headers: $1=NAME $2=READY (1/1) $3=STATUS (Running)
  for selector in "$@"; do
    count=$(oc get pods -n "$ns" -l "$selector" --no-headers 2>/dev/null | awk '$3~/Running/ && $2~/^[0-9]+\/[0-9]+$/ { n++ } END { print n+0 }')
    [[ "${count:-0}" -ge "$min" ]] && break
  done
  if [[ "${count:-0}" -lt "$min" ]]; then
    count=$(oc get pods -n "$ns" --no-headers 2>/dev/null | awk -v pat="$name_pattern" '$3~/Running/ && $2~/^[0-9]+\/[0-9]+$/ && $1~pat { n++ } END { print n+0 }')
  fi
  if [[ "${count:-0}" -ge "$min" ]]; then
    echo "  OK   $name (${count} running)"
  else
    echo "  FAIL $name (expected >= $min running, got ${count:-0})"
    echo "       Hint: oc get pods -n $ns"
    FAILED=1
  fi
}

echo "Monitoring and telemetry checks (namespace: $MONITORING_NAMESPACE)"
[[ "$LENIENT" == true ]] && echo "(lenient: skipping pod-running and Instrumentation checks)"
echo ""

# 1. Namespace
check "Namespace $MONITORING_NAMESPACE exists" "oc get namespace $MONITORING_NAMESPACE"

# 2. TempoMonolithic CR and pods (operator creates deployment; labels vary by operator version)
check "TempoMonolithic data-science-tempomonolithic exists" \
  "oc get tempomonolithic data-science-tempomonolithic -n $MONITORING_NAMESPACE"
if [[ "$LENIENT" != true ]]; then
  check_pods_running "Tempo pods (data-science-tempomonolithic)" "$MONITORING_NAMESPACE" 1 "data-science-tempomonolithic" \
    "app.kubernetes.io/name=tempo" \
    "app.kubernetes.io/instance=${MONITORING_NAMESPACE}.data-science-tempomonolithic"
fi

# 3. OpenTelemetryCollector CR and pods
check "OpenTelemetryCollector data-science-collector exists" \
  "oc get opentelemetrycollector data-science-collector -n $MONITORING_NAMESPACE"
if [[ "$LENIENT" != true ]]; then
  check_pods_running "OTel collector pods (data-science-collector)" "$MONITORING_NAMESPACE" 1 "data-science-collector" \
    "app.kubernetes.io/component=opentelemetry-collector" \
    "app.kubernetes.io/instance=${MONITORING_NAMESPACE}.data-science-collector"
fi

# 4. Tempo and collector services (for trace ingestion and Grafana)
check "Tempo service (tempo-data-science-tempomonolithic)" \
  "oc get svc tempo-data-science-tempomonolithic -n $MONITORING_NAMESPACE"
check "OTel collector headless service (data-science-collector-collector-headless)" \
  "oc get svc data-science-collector-collector-headless -n $MONITORING_NAMESPACE"

# 5. Prometheus (created by OpenShift AI operator when DSCI monitoring is enabled)
if [[ "$SKIP_PROMETHEUS" != true ]]; then
  check "Prometheus service (data-science-monitoringstack-prometheus)" \
    "oc get svc data-science-monitoringstack-prometheus -n $MONITORING_NAMESPACE"
fi

# 6. OpenTelemetry Instrumentation (for inject-python; created by operator; API group may vary)
if [[ "$SKIP_INSTRUMENTATION" != true && "$LENIENT" != true ]]; then
  if oc get instrumentation data-science-instrumentation -n "$MONITORING_NAMESPACE" &>/dev/null || \
     oc get instrumentations.opentelemetry.io data-science-instrumentation -n "$MONITORING_NAMESPACE" &>/dev/null; then
    echo "  OK   Instrumentation data-science-instrumentation exists"
  else
    echo "  FAIL Instrumentation data-science-instrumentation exists"
    echo "       Hint: use --skip-instrumentation if not needed, or ensure the OpenTelemetry Instrumentation CRD is installed"
    FAILED=1
  fi
fi

echo ""
if [[ $FAILED -eq 0 ]]; then
  echo "All checks passed. Monitoring and telemetry are in place."
  exit 0
else
  echo "Some checks failed. Ensure ./scripts/setup-monitoring-for-helm.sh was run and operators have reconciled."
  echo "For pod status: oc get pods -n $MONITORING_NAMESPACE"
  exit 1
fi
