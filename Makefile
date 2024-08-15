.PHONY: lint
lint:
	poetry run pre-commit run --all-files
