RUN_TEST := test -f .env && set -a; . ./.env; set +a; poetry run pytest

.PHONY: lint
lint:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	${RUN_TEST}

.PHONY: test-sw
test-sw:
	${RUN_TEST} -vv --sw --show-capture=no

.PHONY: vendor-claude-agents-plugin
vendor-claude-agents-plugin:
	@test -n "$(PLUGIN_SRC)" || (echo "PLUGIN_SRC is required"; exit 1)
	python scripts/vendor_claude_agents_plugin.py --source "$(PLUGIN_SRC)"
