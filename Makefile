.PHONY: install ingest features train evaluate simulate report test

install:
	pip install -e ".[dev]"

ingest:
	wm ingest --source all

features:
	wm build-features

train:
	wm train --model all

evaluate:
	wm evaluate --split test --baselines

simulate:
	wm simulate --runs 100000

report:
	wm report --out reports/wc2026_report.html

test:
	pytest -v

pipeline: ingest features train evaluate simulate report
