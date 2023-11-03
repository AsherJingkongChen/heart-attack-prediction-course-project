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
    dots = '.' * (1 + i // 10) + ' ' * (4 - i // 10)
    log.info(f'Evaluating {dots}\033[A')
    i = (i + 1) % 30
    sleep(1 / 20)

def main():
  from multiprocessing import cpu_count, Process
  from numpy import mean
  import pandas
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    GridSearchCV,
  )
  from sklearn.preprocessing import LabelEncoder
  from sys import argv, executable
  from xgboost import XGBClassifier
  pandas.options.mode.chained_assignment = None

  # check arguments
  argc = len(argv)
  if argc != 2:
    raise TypeError(f'\
  Program expected exact 2 arguments, got {argc}. \
  Usage: {executable} {argv[0]} <data_path>')

  # get arguments
  data_path = argv[1]

  # read data
  data = pandas.read_csv(data_path)
  log.info(f'Loaded dataset {data_path} to memory')

  # remove irrelevant features
  irrevalent_features = [
    'BMI',
    'DifficultyDressingBathing',
    'HighRiskLastYear',
  ]
  data.drop(columns=irrevalent_features, inplace=True)
  log.info('Removed irrelevant features')

  # remove rows with missing values
  data.dropna(inplace=True)
  log.info('Removed rows with missing values')

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
  y = pandas.Series(LabelEncoder().fit_transform(y))
  log.info('Transformed values')
  log.debug(f'transformed x:\n{x.head(10)}')
  log.debug(f'transformed y:\n{y.head(10)}')

  # define model
  # [todo] GridSearchCV tuning
  model = XGBClassifier(
    colsample_bynode=0.6,
    colsample_bytree=0.6,
    gamma=0.001,
    learning_rate=0.1,
    max_delta_step=1,
    max_depth=6,
    min_child_weight=60,
    n_estimators=100,
    n_jobs=max(1, cpu_count() - 1),
    num_parallel_tree=6,
    random_state=6,
    reg_alpha=0.001,
    scale_pos_weight=6,
    subsample=0.6,
  )

  log.debug(f'model:\n{model}')

  # check auc
  log.info('Started model evaluation')
  proc_timer = Process(target=timer_loop)
  proc_timer.start()
  scores = cross_val_score(
    model, x, y,
    scoring='roc_auc',
    cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=6),
    n_jobs=1,
  )
  proc_timer.terminate()
  log.info('Terminated model evaluation')
  log.info(f'Average AUC: {mean(scores)}')
  log.debug(f'AUCs:\n{scores}')

if __name__ == '__main__':
  main()
