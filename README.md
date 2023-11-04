# Heart Attack Prediction

## Datasets

### Indicators of Heart Disease (2022 UPDATE)
- source: [link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/)
- profile: [link](./logs/profile_of_heart_2022_with_nans.log) (produced by [data profiler](./src/data_profiler.py))

## Models

### XGBoost (E<u>x</u>treme <u>G</u>radient <u>Boost</u>ing)
- Python package: [link](https://xgboost.readthedocs.io/en/stable/python/index.html)
- Use [`XGBClassfier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) in its Python package.
- evaluation: [link](./src/xgboost_model_evaluator.py)
- logging: [link](./logs/run_xgboost_model_evaluator.py.log)

## Environments
- Python 3.9 with PIP
- Windows, macOS (x86_64 & ARM64)
