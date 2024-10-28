.PHONY: isort
isort:
	poetry run isort ./exhbma ./tests

.PHONY: black
black:
	poetry run black ./exhbma ./tests

.PHONY: format
format: isort black

.PHONY: flake8
flake8:
	poetry run flake8 ./exhbma ./tests

.PHONY: mypy
mypy:
	poetry run mypy ./exhbma ./tests

.PHONY: lint
lint: flake8 mypy

.PHONY: test
test:
	poetry run pytest ./tests

.PHONY: test-full
test-full:
	poetry run pytest ./tests --log-cli-level=INFO --full

.PHONY: test-full-force-update
test-full-force-update:
	poetry run pytest ./tests --log-cli-level=INFO --full --force-update
