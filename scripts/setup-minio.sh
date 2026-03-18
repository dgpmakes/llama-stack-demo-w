#!/usr/bin/env bash
# Creates an Argo CD Application for MinIO (rhoai-gitops components/minio).
# Run as cluster-admin. The application syncs MinIO from the GitOps repo into
# the configured namespace.
#
# Usage: setup-minio.sh [--dry-run]
# Env:   MINIO_NAMESPACE      Target namespace (default: minio)
#        MINIO_SERVICE_NAME   MinIO service name (default: minio)
#        MINIO_ADMIN_USERNAME Admin username (default: minio)
#        MINIO_ADMIN_PASSWORD Admin password (default: minio123)
#        CLUSTER_DOMAIN       OpenShift cluster domain (default: from ingresses.config.openshift.io)

set -euo pipefail

ARGOCD_NS="openshift-gitops"
MINIO_NAMESPACE="${MINIO_NAMESPACE:-minio}"
MINIO_SERVICE_NAME="${MINIO_SERVICE_NAME:-minio}"
MINIO_ADMIN_USERNAME="${MINIO_ADMIN_USERNAME:-minio}"
MINIO_ADMIN_PASSWORD="${MINIO_ADMIN_PASSWORD:-minio123}"

usage() {
  echo "Usage: $0 [--dry-run]" >&2
  echo "  --dry-run  Preview the Argo CD Application without applying." >&2
  echo "" >&2
  echo "Optional env: MINIO_NAMESPACE, MINIO_SERVICE_NAME, MINIO_ADMIN_USERNAME, MINIO_ADMIN_PASSWORD, CLUSTER_DOMAIN" >&2
  exit 1
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Error: unknown option $1" >&2; usage ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! command -v oc &>/dev/null; then
    echo "Error: oc (OpenShift CLI) is required." >&2
    exit 2
  fi
  if ! oc whoami &>/dev/null; then
    echo "Error: you are not logged into OpenShift. Run 'oc login' and try again." >&2
    exit 2
  fi
fi

if [[ -z "${CLUSTER_DOMAIN:-}" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    CLUSTER_DOMAIN="cluster.example.com"
  else
    CLUSTER_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')
  fi
fi

echo "Creating Argo CD Application for MinIO..."
echo "  namespace: $MINIO_NAMESPACE"
echo "  service:   $MINIO_SERVICE_NAME"
echo "  clusterDomain: $CLUSTER_DOMAIN"

APPLICATION_YAML=$(cat <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  labels:
    app: minio
  name: minio
  namespace: ${ARGOCD_NS}
spec:
  destination:
    server: 'https://kubernetes.default.svc'
  project: default
  source:
    path: components/minio
    repoURL: https://github.com/alvarolop/rhoai-gitops.git
    targetRevision: main
    helm:
      values: |
        clusterDomain: ${CLUSTER_DOMAIN}
        namespace: ${MINIO_NAMESPACE}
        service:
          name: ${MINIO_SERVICE_NAME}
        adminUser:
          username: ${MINIO_ADMIN_USERNAME}
          password: ${MINIO_ADMIN_PASSWORD}
  syncPolicy:
    automated:
      prune: false
      selfHeal: false
    syncOptions:
      - CreateNamespace=true
EOF
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "Application YAML that would be applied:"
  echo "---"
  echo "$APPLICATION_YAML"
  echo ""
  echo "Dry-run complete. Run without --dry-run to apply."
  exit 0
fi

echo "$APPLICATION_YAML" | run oc apply -f -
echo "Done. MinIO Argo CD Application created in ${ARGOCD_NS}."
