# Heart Attack Prediction

# Bootstrap
*nix os:
```shell
python3 -m pip install -r requirements.txt
python3 src/data_profiler.py data/heart_2022_no_nans.csv
python3 src/xgboost_model_evaluator.py data/heart_2022_no_nans.csv
```

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

### Three Layers NN ML model
- Python Tensorflow + Keras
- notebook: [link](./notebooks/three_layers_nn_model_eval.ipynb)

## Environments
- Python 3.9 with PIP
- Windows, macOS (x86_64 & ARM64)
- Google Colab
