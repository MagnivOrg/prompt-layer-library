#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=plugins/trace/hooks/lib.sh
source "$SCRIPT_DIR/lib.sh"

tracing_enabled || exit 0
check_requirements || exit 0

input="$(cat)"
result="$(printf '%s' "$input" | python3 "$SCRIPT_DIR/py/cli.py" stop-hook)"
IFS=$'\t' read -r session_id status <<<"$result"
[[ -z "$session_id" ]] && exit 0

if [[ "$status" == "missing_transcript" ]]; then
	log "WARN" "Stop missing transcript session_id=$session_id"
fi

log "INFO" "Stop finalized session_id=$session_id"
