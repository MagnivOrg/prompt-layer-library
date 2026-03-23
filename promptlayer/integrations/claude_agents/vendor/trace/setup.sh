#!/bin/bash
set -euo pipefail

echo "PromptLayer Claude Code tracing setup"
echo "====================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$SCRIPT_DIR/hooks"
DEFAULT_ENDPOINT="https://api.promptlayer.com/v1/traces"

install_hint() {
	local cmd="$1"
	if [[ "$OSTYPE" == "darwin"* ]]; then
		echo "  Install with: brew install $cmd"
	else
		echo "  Install with: sudo apt-get install $cmd"
	fi
}

load_env_key() {
	local dir="$PWD"
	while [[ "$dir" != "/" ]]; do
		if [[ -f "$dir/.env" ]]; then
			while IFS= read -r line || [[ -n "$line" ]]; do
				[[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
				if [[ "$line" =~ ^[[:space:]]*PROMPTLAYER_API_KEY= ]]; then
					local value
					value="${line#*=}"
					value="${value%\"}"
					value="${value#\"}"
					value="${value%\'}"
					value="${value#\'}"
					if [[ -n "$value" ]]; then
						echo "$value"
						return 0
					fi
				fi
			done <"$dir/.env"
		fi
		dir="$(dirname "$dir")"
	done
	return 1
}

test_endpoint() {
	local endpoint="$1"
	local api_key="$2"
	local status
	status="$(python3 "$HOOKS_DIR/py/cli.py" probe-endpoint "$endpoint" "$api_key")"

	if [[ "$status" == "000" || -z "$status" ]]; then
		echo "WARN: Could not reach endpoint: $endpoint"
		echo "      Check network/DNS/firewall settings."
		return 0
	fi

	if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
		echo "OK: Endpoint reachable and accepted probe payload (HTTP $status)."
		return 0
	fi

	if [[ "$status" =~ ^4[0-9][0-9]$ ]]; then
		if [[ "$status" == "401" || "$status" == "403" ]]; then
			echo "WARN: Endpoint reachable but API key may be invalid (HTTP $status)."
		else
			echo "WARN: Endpoint reachable but probe rejected (HTTP $status)."
		fi
		return 0
	fi

	if [[ "$status" =~ ^5[0-9][0-9]$ ]]; then
		echo "WARN: Endpoint reachable but returned server error (HTTP $status)."
		return 0
	fi

	echo "WARN: Endpoint check returned unexpected status: $status"
}

for hook in lib.sh session_start.sh user_prompt_submit.sh post_tool_use.sh stop_hook.sh session_end.sh hooks.json; do
	if [[ ! -f "$HOOKS_DIR/$hook" ]]; then
		echo "Error: missing plugin file: $HOOKS_DIR/$hook"
		exit 1
	fi
done

for py_file in cli.py context.py handlers.py otlp.py settings.py state.py stop_parser.py traceparent.py; do
	if [[ ! -f "$HOOKS_DIR/py/$py_file" ]]; then
		echo "Error: missing plugin file: $HOOKS_DIR/py/$py_file"
		exit 1
	fi
done

if ! command -v python3 >/dev/null 2>&1; then
	echo "Error: missing required command: python3"
	install_hint "python3"
	exit 1
fi

default_key="${PROMPTLAYER_API_KEY:-}"
if [[ -z "$default_key" ]]; then
	if env_key="$(load_env_key 2>/dev/null)"; then
		default_key="$env_key"
		echo "Found PROMPTLAYER_API_KEY in a parent .env file."
	fi
fi

if [[ -n "$default_key" ]]; then
	echo "Press Enter to use detected PROMPTLAYER_API_KEY, or type a different key:"
	read -r -s -p "> " input_key
	echo ""
	api_key="${input_key:-$default_key}"
else
	echo "You can find or create an API key at: https://dashboard.promptlayer.com"
	read -r -s -p "Enter PROMPTLAYER_API_KEY (input hidden): " input_key
	echo ""
	api_key="$input_key"
fi

if [[ -z "$api_key" ]]; then
	echo "Error: PROMPTLAYER_API_KEY is required."
	exit 1
fi

if [[ ! "$api_key" =~ ^pl_ ]]; then
	echo "Warning: API key does not start with 'pl_'. Continuing anyway."
fi

read -r -p "OTLP endpoint [$DEFAULT_ENDPOINT]: " input_endpoint
endpoint="${input_endpoint:-$DEFAULT_ENDPOINT}"
if [[ "$endpoint" == *"/otel/v1/traces" ]]; then
	endpoint="${endpoint%/otel/v1/traces}/v1/traces"
	echo "Updated endpoint to new ingestion route: $endpoint"
fi

if [[ ! "$endpoint" =~ ^https?:// ]]; then
	echo "Error: endpoint must start with http:// or https://"
	exit 1
fi

read -r -p "Enable debug logging? (y/N): " input_debug
if [[ "$input_debug" =~ ^[Yy]$ ]]; then
	debug="true"
else
	debug="false"
fi

mkdir -p "$HOME/.claude"
settings_file="$HOME/.claude/settings.json"

if ! python3 "$HOOKS_DIR/py/cli.py" write-settings-env "$settings_file" "$api_key" "$endpoint" "$debug" >/dev/null 2>&1; then
	echo "Error: $settings_file exists but is not valid JSON."
	echo "Fix or remove it, then rerun setup."
	exit 1
fi

chmod 600 "$settings_file" 2>/dev/null || true

echo ""
echo "Configuration written to $settings_file"
echo "Testing endpoint connectivity..."
test_endpoint "$endpoint" "$api_key"
echo ""
echo "Setup complete."
echo "Next:"
echo "  1. Start Claude Code in this directory: claude"
echo "  2. Run one prompt and tool call"
echo "  3. View your traces at: https://dashboard.promptlayer.com"
echo ""
echo "Debug logs: tail -f ~/.claude/state/promptlayer_hook.log"
