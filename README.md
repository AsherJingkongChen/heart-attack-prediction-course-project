# Heart Attack Prediction

## Note

### Please install the complete data file from [Kaggle.com](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/)
> Sorry but the file is too big

## Datasets

### Indicators of Heart Disease (2022 UPDATE)
- path: [data/heart_2022_no_nans.csv](./data/heart_2022_no_nans.csv)
- source: [link](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/)
- profile: [link](./logs/profile_of_heart_2022_no_nans.log) (produced by [data profiler](./src/data_profiler.py))

## Models

### XGBoost (E[x](#)treme [G](#)radient [Boost](#)ing)
- Python package: [link](https://xgboost.readthedocs.io/en/stable/python/index.html)
- Use [`XGBClassfier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) in the Python package.
- evaluation: [link](./src/xgboost_model_evaluator.py)
- logging: [link](./logs/run_xgboost_model_evaluator.py.log)

## Environments
- Pyt hon 3.9 with PIP
- Windows, macOS (x86_64 & ARM64)
