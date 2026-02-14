install:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset

train:
	python -m src.models.train_model

evaluate:
	python -m src.models.evaluate_model

api:
	venv\Scripts\python.exe -m uvicorn src.api.app:app --reload


test:
	venv/Scripts/python.exe -m pytest


