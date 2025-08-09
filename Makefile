.PHONY: train run api test lint format docker-build docker-run

train:
	python src/get_data.py
	python src/train.py

run:
	uvicorn src.app:app --host 0.0.0.0 --port 5000

api: run

test:
	pytest -q

lint:
	flake8 src tests

docker-build:
	docker build -t mlops-app:local .

docker-run:
	docker run --rm -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts mlops-app:local
