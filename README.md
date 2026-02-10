# MLOps NLP Ticket Classifier

A beginner-friendly MLOps project that classifies customer support tickets into business categories such as billing, delivery, refunds, technical support, and account issues.

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

### 2. Train Model
```bash
python src/models/train_model.py

### 3. Run Tests
```bash
pytest
