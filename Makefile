.PHONY: install data train evaluate predict benchmark ablation figures test lint clean

MODEL ?= hist_gb

install:
	pip install -r requirements.txt

data:
	python prepare_data.py

train:
	python train.py --model $(MODEL)

evaluate:
	python evaluate.py --model $(MODEL)

predict:
	python predict.py examples/listing.json --model $(MODEL)

benchmark:
	python benchmark.py --models ridge hist_gb random_forest mlp

ablation:
	python benchmark.py --models ridge hist_gb --feature-set no_performance
	python benchmark.py --models ridge hist_gb --feature-set pre_booking

figures:
	python make_figures.py --stage all

test:
	pytest -v --cov=src

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
