# MLOps NLP Ticket Classifier

A beginner-friendly MLOps project that classifies customer support tickets into business categories such as billing, delivery, refunds, technical support, and account issues. MLOps

#Machine Learning #DeepLearning #Python #scikit-learn #FastAPI #NLP #TextClassification #Data Science

## Project Structure
- `data/` - synthetic dataset storage
- `src/` - source code
- `models/` - trained models
- `tests/` - unit tests
- `docs/` - documentation

## How to Run
### 1. Generate Dataset
```bash
python src/data/make_dataset.py
```
### 2. Train Model
```bash
python src/models/train_model.py
```

### 3. Run Tests
```bash
pythom -m pytest
```
## Model Evaluation

After training, the model is evaluated on a test split (20% of data).

- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion Matrix generated
- Full evaluation report saved in `/reports/evaluation_report.txt`

Example:
Classification Report
====================
                   precision    recall  f1-score   support

          account       1.00      1.00      1.00        27
          billing       1.00      1.00      1.00        19
         delivery       1.00      1.00      1.00        22

## Run API

Start the API server:
```bash
uvicorn src.api.app:app --reload
```

### Health check:

[Health Check](http://127.0.0.1:8000/health)

### Swagger docs:

[Doc](http://127.0.0.1:8000/docs)
