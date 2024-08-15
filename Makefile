.PHONY: lint
lint:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	poetry run pytest
