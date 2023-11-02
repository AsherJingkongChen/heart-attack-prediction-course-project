"""
Evaluate an XGBoost model on a heart disease dataset
using repeated stratified k-fold cross-validation.
"""

import logging as log
from time import sleep
log.basicConfig(
  level=log.INFO,
  format='%(levelname)s\t%(asctime)s.%(msecs)03d\t%(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
)

def timer_loop():
  i = 0
  while True:
    dots = '.' * (1 + i // 15) + ' ' * (4 - i // 15)
    log.info(f'Evaluating {dots}\033[A')
    i = (i + 1) % 45
    sleep(1 / 30)

if __name__ == '__main__':
  from multiprocessing import Process
  from numpy import mean
  import pandas as pd
  from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
  )
  from sklearn.preprocessing import LabelEncoder
  import xgboost as xgb

  pd.options.mode.chained_assignment = None

  # read data
  data_path = 'data/heart_2022_with_nans.csv'
  data = pd.read_csv(data_path)
  log.info(f'Loaded dataset {data_path} to memory.')

  # remove irrelevant features
  irrevalent_features = [
    'BMI',
    'DifficultyDressingBathing',
    'HIVTesting',
    'HighRiskLastYear',
    'State',
  ]
  data.drop(columns=irrevalent_features, inplace=True)
  log.info('Removed irrelevant features.')

  # remove rows with missing values
  data.dropna(inplace=True)
  log.info('Removed rows with missing values.')

  # define target and features
  target = 'HadHeartAttack'
  x = data.drop(columns=target)
  y = data[target]
  log.debug(f'x:\n{x.head(10)}')
  log.debug(f'y:\n{y.head(10)}')

  # transform values to numeric
  for col in x.columns:
    if x[col].dtype == 'object':
      x[col] = LabelEncoder().fit_transform(x[col])
  y = pd.Series(LabelEncoder().fit_transform(y))
  log.info('Transformed values.')
  log.debug(f'transformed x:\n{x.head(10)}')
  log.debug(f'transformed y:\n{y.head(10)}')

  # create model
  model = xgb.XGBClassifier(
    colsample_bynode=0.6,
    colsample_bytree=0.6,
    gamma=0.001,
    learning_rate=0.1,
    max_delta_step=1,
    max_depth=6,
    min_child_weight=60,
    n_estimators=100,
    n_jobs=1,
    num_parallel_tree=6,
    random_state=6,
    reg_alpha=0.001,
    scale_pos_weight=6,
    subsample=0.6,
  )

  log.debug(f'model:\n{model}')

  # check auc
  log.info('Started model evaluation.')
  proc_timer = Process(target=timer_loop)
  proc_timer.start()
  scores = cross_val_score(
    model, x, y,
    scoring='roc_auc',
    cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=6),
    n_jobs=6,
  )
  proc_timer.terminate()
  log.info('Terminated model evaluation.')
  log.info(f'Average AUC: {mean(scores)}')
  log.debug(f'AUCs:\n{scores}')
