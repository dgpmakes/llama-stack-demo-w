#!/usr/bin/env bash
# Creates workshop environment: generates htpasswd file (admin applies manually),
# creates projects, workshop group with permissions, and optionally assigns nodes.
#
# Usage: workshop-setup.sh [--dry-run] [--no-assign] <number_of_users> [password]
# Env:   CUSTOM_PROJECT  Project name prefix (default: llama-stack-demo)
#
# Step 1: Always runs setup-htpasswd-oauth.sh in dry-run mode. Generates htpasswd
#         file and prints instructions for the Administrator to apply it manually.
# Step 2: Creates projects (${CUSTOM_PROJECT}-user1, ...) with labels.
# Step 3: Creates group "workshop", adds users, grants admin per project (idempotent).
# Step 4: Runs setup-monitoring.sh, setup-hardware-profile.sh, setup-minio.sh, setup-mlflow.sh,
#         setup-rbac.sh, and setup-grafana-proxy-rbac.sh.
# Step 5: Runs assign-nodes-to-users.sh unless --no-assign is passed.
#
# Projects: ${CUSTOM_PROJECT}-user1, ${CUSTOM_PROJECT}-user2, ...
# Labels:   modelmesh-enabled=false opendatahub.io/dashboard=true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CUSTOM_PROJECT="${CUSTOM_PROJECT:-llama-stack-demo}"
GROUP_NAME="${GROUP_NAME:-workshop}"
HTPASSWD_OUTPUT="${HTPASSWD_OUTPUT:-$REPO_ROOT/htpasswd.workshop}"

usage() {
  echo "Usage: $0 [--dry-run] [--no-assign] <number_of_users> [password]" >&2
  echo "  --dry-run       Preview all actions without making changes." >&2
  echo "  --no-assign     Skip node assignment (assign-nodes-to-users.sh)." >&2
  echo "  number_of_users Number of users (user1..userN) and projects." >&2
  echo "  password        Optional. Password for all users (default: generated)." >&2
  echo "" >&2
  echo "Optional env: CUSTOM_PROJECT (default: llama-stack-demo), HTPASSWD_OUTPUT (default: htpasswd.workshop), INSTANCE_TYPE (default: g5.2xlarge)" >&2
  exit 1
}

DRY_RUN=0
NO_ASSIGN=0
NUM_USERS=""
PASSWORD_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --no-assign) NO_ASSIGN=1; shift ;;
    *)
      if [[ -z "$NUM_USERS" ]]; then
        NUM_USERS="$1"
      elif [[ -z "$PASSWORD_ARG" ]]; then
        PASSWORD_ARG="$1"
      fi
      shift
      ;;
  esac
done

if [[ -z "$NUM_USERS" ]]; then
  usage
fi

if ! [[ "$NUM_USERS" =~ ^[0-9]+$ ]] || [[ "$NUM_USERS" -lt 1 ]]; then
  echo "Error: number_of_users must be a positive integer." >&2
  usage
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! command -v oc &>/dev/null; then
    echo "Error: oc (OpenShift CLI) is required." >&2
    exit 2
  fi
  if ! oc whoami &>/dev/null; then
    echo "Error: you are not logged into OpenShift. Run 'oc login' and try again." >&2
    exit 2
  fi

  # Check required permissions (cluster-admin or equivalent)
  MISSING_PERMS=()
  oc auth can-i create projectrequest &>/dev/null || MISSING_PERMS+=("create projectrequest")
  oc auth can-i create groups &>/dev/null || MISSING_PERMS+=("create groups")
  oc auth can-i create rolebindings --all-namespaces &>/dev/null || MISSING_PERMS+=("create rolebindings")
  oc auth can-i create clusterroles &>/dev/null || MISSING_PERMS+=("create clusterroles")
  oc auth can-i create clusterrolebindings &>/dev/null || MISSING_PERMS+=("create clusterrolebindings")
  if [[ "$NO_ASSIGN" -eq 0 ]]; then
    oc auth can-i patch nodes &>/dev/null || MISSING_PERMS+=("patch nodes")
  fi
  if [[ ${#MISSING_PERMS[@]} -gt 0 ]]; then
    echo "Error: insufficient permissions. Missing: ${MISSING_PERMS[*]}" >&2
    echo "This script requires cluster-admin or equivalent. Run 'oc login' as a cluster administrator." >&2
    exit 2
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== DRY RUN: No changes will be made ==="
  echo ""
fi

# -----------------------------------------------------------------------------
# Step 1: Generate htpasswd file (always dry-run) and give admin instructions
# -----------------------------------------------------------------------------
echo "Step 1: Generating htpasswd file for user1..user${NUM_USERS}..."
export HTPASSWD_OUTPUT
htpasswd_output=$("$SCRIPT_DIR/setup-htpasswd-oauth.sh" --dry-run --silent "$NUM_USERS" ${PASSWORD_ARG:+"$PASSWORD_ARG"} 2>/dev/null)
eval "$htpasswd_output"
DISPLAY_PASSWORD="${PASSWORD_ARG:-${HTPASSWD_PASSWORD:-}}"

echo ""
echo "--- Administrator instructions ---"
echo "The htpasswd file has been written to: ${HTPASSWD_OUTPUT}"
echo ""
echo "To configure OAuth with these users, run (as cluster-admin):"
echo ""
echo "  1. Create the secret:"
echo "     oc create secret generic htpasswd-secret --from-file=htpasswd=${HTPASSWD_OUTPUT} -n openshift-config --dry-run=client -o yaml | oc apply -f -"
echo ""
echo "  2. Add or update the HTPasswd identity provider in OAuth:"
echo "     oc edit oauth cluster"
echo "     (Add/ensure an identityProvider with type: HTPasswd, htpasswd.fileData.name: htpasswd-secret)"
echo ""
echo "  Or run setup-htpasswd-oauth.sh without --dry-run to apply automatically:"
echo "     ./scripts/setup-htpasswd-oauth.sh $NUM_USERS <password>"
[[ -n "$DISPLAY_PASSWORD" ]] && echo "     Password: ${DISPLAY_PASSWORD}"
echo ""
echo "---"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Step 2-5 skipped (dry-run)."
  echo ""
  echo "=== Dry-run summary ==="
  echo "  Users:     user1..user${NUM_USERS}"
  echo "  Htpasswd:  ${HTPASSWD_OUTPUT}"
  [[ -n "$DISPLAY_PASSWORD" ]] && echo "  Password:  ${DISPLAY_PASSWORD}" || true
  echo ""
  echo "Run without --dry-run to create projects, group, and assign nodes."
  exit 0
fi

# -----------------------------------------------------------------------------
# Step 2: Create projects and label them
# -----------------------------------------------------------------------------
echo "Step 2: Creating projects and applying labels..."
for (( i = 1; i <= NUM_USERS; i++ )); do
  PROJECT="${CUSTOM_PROJECT}-user${i}"
  if oc get project "$PROJECT" &>/dev/null; then
    echo "  Project ${PROJECT} already exists, updating labels..."
  else
    echo "  Creating project ${PROJECT}..."
    oc new-project "$PROJECT" >/dev/null
  fi
  oc label namespace "$PROJECT" modelmesh-enabled=false opendatahub.io/dashboard=true --overwrite
  echo "  Labeled ${PROJECT}"
done

echo ""

# -----------------------------------------------------------------------------
# Step 3: Create group and assign permissions (idempotent)
# -----------------------------------------------------------------------------
export CUSTOM_PROJECT GROUP_NAME
"$SCRIPT_DIR/create-workshop-group.sh" "$NUM_USERS"

echo ""

# -----------------------------------------------------------------------------
# Step 4: Setup monitoring, MLflow, and Grafana proxy RBAC
# -----------------------------------------------------------------------------
echo "Step 4: Setting up monitoring, hardware profile, MinIO, MLflow, configmap-patcher RBAC, and Grafana proxy RBAC..."
export CUSTOM_PROJECT
"$SCRIPT_DIR/setup-monitoring.sh"
"$SCRIPT_DIR/setup-hardware-profile.sh"
"$SCRIPT_DIR/setup-minio.sh"
"$SCRIPT_DIR/setup-mlflow.sh" "$NUM_USERS"
"$SCRIPT_DIR/setup-rbac.sh" "$NUM_USERS"
"$SCRIPT_DIR/setup-grafana-proxy-rbac.sh" "$NUM_USERS"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Assign nodes to users (unless --no-assign)
# -----------------------------------------------------------------------------
if [[ "$NO_ASSIGN" -eq 0 ]]; then
  echo "Step 5: Assigning nodes to users..."
  export CUSTOM_LABEL_PREFIX="$CUSTOM_PROJECT"
  INSTANCE_TYPE="${INSTANCE_TYPE:-g5.2xlarge}"
  "$SCRIPT_DIR/assign-nodes-to-users.sh" --summary "$NUM_USERS" "$INSTANCE_TYPE"
else
  echo "Step 5: Skipped (--no-assign)."
fi

echo ""
echo "=== Workshop setup complete ==="
echo ""
echo "Summary:"
echo "  Users:     user1..user${NUM_USERS}"
echo "  Projects:  ${CUSTOM_PROJECT}-user1..user${NUM_USERS}"
echo "  Group:     ${GROUP_NAME}"
echo "  Htpasswd:  ${HTPASSWD_OUTPUT}"
[[ -n "$DISPLAY_PASSWORD" ]] && echo "  Password:  ${DISPLAY_PASSWORD}" || true
echo ""
echo "Next: Apply htpasswd to OAuth (see instructions above), then each user runs:"
echo "  PROJECT=\"${CUSTOM_PROJECT}-user<N>\""
echo "  helm install llama-stack-demo helm/ -f helm/values-workshop.yaml --set assigned=\"\${PROJECT}\" --namespace \${PROJECT} --timeout 20m"
