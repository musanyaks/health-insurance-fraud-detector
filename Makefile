.PHONY: install data features train test api shiny docker-up all

install:
	pip install -e ".[dev]"

data:
	python synthetic/generate_claims.py

features:
	python etl/feature_pipeline.py

train: data
	python fraud_engine/supervised/train_logistic.py
	python fraud_engine/supervised/train_xgboost.py

test:
	python -m pytest tests/ -v

api:
	uvicorn api.main:app --reload --port 8000

shiny:
	Rscript -e "shiny::runApp('dashboard/shiny', launch.browser = TRUE)"

docker-up:
	docker compose up --build

all: data features train test