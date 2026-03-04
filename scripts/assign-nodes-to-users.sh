#!/usr/bin/env bash
# Assigns one node per user by labelling nodes of a given instance type.
# Portable: macOS (Bash 3.x) and Linux (Bash 4+).
# Nodes are labelled with ${CUSTOM_LABEL}=${CUSTOM_LABEL_PREFIX}-user${i}
# (default: assigned=llama-stack-demo-user1, etc.)
#
# Usage: assign-nodes-to-users.sh <number_of_users> [instance_type]
# Env:   CUSTOM_LABEL       label key (default: assigned)
#        CUSTOM_LABEL_PREFIX  label value prefix (default: llama-stack-demo)

set -euo pipefail

CUSTOM_LABEL="${CUSTOM_LABEL:-assigned}"
CUSTOM_LABEL_PREFIX="${CUSTOM_LABEL_PREFIX:-llama-stack-demo}"

usage() {
  echo "Usage: $0 <number_of_users> [instance_type]" >&2
  echo "  number_of_users  Number of users (nodes to assign)." >&2
  echo "  instance_type    Node instance type (default: g5.4xlarge)." >&2
  echo "" >&2
  echo "Optional env: CUSTOM_LABEL (default: assigned), CUSTOM_LABEL_PREFIX (default: llama-stack-demo)" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

NUM_USERS="$1"
INSTANCE_TYPE="${2:-g5.4xlarge}"
USERS_WITH_NODES=()

if ! [[ "$NUM_USERS" =~ ^[0-9]+$ ]] || [[ "$NUM_USERS" -lt 1 ]]; then
  echo "Error: number_of_users must be a positive integer." >&2
  usage
fi

# Get nodes of this instance type that do NOT already have CUSTOM_LABEL
# (using jq to filter out nodes that have the label set)
NODES_JSON=$(oc get nodes -l "node.kubernetes.io/instance-type=${INSTANCE_TYPE}" -o json 2>/dev/null) || {
  echo "Error: failed to get nodes (check 'oc' and cluster access)." >&2
  exit 2
}

# Show existing assignments (nodes that already have CUSTOM_LABEL set)
# and collect which user numbers already have a node (e.g. llama-stack-demo-user1 -> 1)
ASSIGNED_ENTRIES=$(echo "$NODES_JSON" | jq -r --arg key "$CUSTOM_LABEL" '
  .items[] | select(.metadata.labels[$key] != null and .metadata.labels[$key] != "") | "\(.metadata.name) \(.metadata.labels[$key])"
')
if [[ -n "$ASSIGNED_ENTRIES" ]]; then
  echo "Existing assignments (instance type ${INSTANCE_TYPE}):"
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      echo "  ${line%% *} is assigned to ${line#* }"
      # Extract user number from value (e.g. llama-stack-demo-user1 -> 1)
      val="${line#* }"
      if [[ "$val" =~ -user([0-9]+)$ ]]; then
        USERS_WITH_NODES+=("${BASH_REMATCH[1]}")
      fi
    fi
  done <<< "$ASSIGNED_ENTRIES"
fi

has_node() {
  local u="$1"
  local i
  # Safe for empty array with set -u (bash 3.x / macOS)
  for i in "${USERS_WITH_NODES[@]+"${USERS_WITH_NODES[@]}"}"; do
    [[ "$i" == "$u" ]] && return 0
  done
  return 1
}

# Build list of unassigned node names (nodes without CUSTOM_LABEL)
UNASSIGNED_NODES=()
while IFS= read -r name; do
  [[ -n "$name" ]] && UNASSIGNED_NODES+=("$name")
done < <(echo "$NODES_JSON" | jq -r --arg key "$CUSTOM_LABEL" '
  .items[] | select(.metadata.labels[$key] == null or .metadata.labels[$key] == "") | .metadata.name
')

AVAILABLE="${#UNASSIGNED_NODES[@]}"
ALREADY_ASSIGNED="${#USERS_WITH_NODES[@]}"
MISSING=$((NUM_USERS - ALREADY_ASSIGNED))

if [[ "$AVAILABLE" -lt "$MISSING" ]]; then
  echo "Warning: Not enough unassigned nodes. Need ${NUM_USERS}, ${ALREADY_ASSIGNED} assigned, ${AVAILABLE} available, ${MISSING} missing." >&2
  for (( u = 1; u <= NUM_USERS; u++ )); do
    if ! has_node "$u"; then
      echo "  No node was assigned to user${u}." >&2
    fi
  done
fi

ASSIGNED=0
for (( i = 1; i <= NUM_USERS; i++ )); do
  idx=$((i - 1))
  if [[ "$idx" -ge "$AVAILABLE" ]]; then
    break
  fi
  NODE="${UNASSIGNED_NODES[$idx]}"
  VALUE="${CUSTOM_LABEL_PREFIX}-user${i}"
  if oc label node "$NODE" "${CUSTOM_LABEL}=${VALUE}" --overwrite 2>/dev/null; then
    (( ASSIGNED++ )) || true
    echo "Assigned $NODE -> ${CUSTOM_LABEL}=${VALUE}"
  else
    echo "Warning: failed to label node $NODE" >&2
  fi
done

echo "Successfully assigned ${ASSIGNED} node(s)."
