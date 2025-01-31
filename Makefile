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
