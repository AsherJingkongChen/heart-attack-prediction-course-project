"""
Evaluate an XGBoost model on a heart disease dataset
using stratified k-fold cross-validation.
"""

import logging as log
from numpy import mean
import pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
  StratifiedKFold,
  cross_val_score,
  train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sys import argv, executable
from xgboost import XGBClassifier

log.basicConfig(
  level=log.INFO,
  format='%(levelname)s\t%(asctime)s.%(msecs)03d\t%(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
)

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
  'MentalHealthDays',
  'PhysicalHealthDays',
  'State',
]
data.drop(columns=irrevalent_features, inplace=True)
log.info('Removed irrelevant features')

# remove rows with missing values
data.dropna(inplace=True)
log.info('Removed rows with missing values')

# transform values to numeric
for col in data.columns:
  if data[col].dtype == 'object':
    data[col] = LabelEncoder().fit_transform(data[col])

# split target and features
target = 'HadHeartAttack'
x = data.drop(columns=target)
y = data[target]
# x = StandardScaler().fit_transform(x, y)

log.info('Transformed values')

# define model
model = XGBClassifier(
  n_jobs=1,
  random_state=6,
  n_estimators=500,
  learning_rate=0.01,
  max_depth=8,
  min_child_weight=42,
  scale_pos_weight=5,
  subsample=0.6,
  reg_alpha=0.0,
  reg_lambda=3.75,
  gamma=0.5,
)
log.debug(f'model:\n{model}')

# cross validation
log.info('Started model cross validation')
scores = cross_val_score(
  model, x, y,
  scoring='roc_auc',
  cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=6),
  n_jobs=4,
)
log.info('Terminated model cross validation')
log.debug(f'AUCs:\n{scores}')
log.info(f'Average AUC: {mean(scores)}')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
model = XGBClassifier(
  n_jobs=1,
  random_state=6,
  n_estimators=500,
  learning_rate=0.01,
  max_depth=8,
  min_child_weight=42,
  scale_pos_weight=5,
  subsample=0.6,
  reg_alpha=0.0,
  reg_lambda=3.75,
  gamma=0.5,
)
model.fit(x_train, y_train)

# evaluate model
log.info('Started model standalone evaluation')
y_pred = model.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_pred)

log.info('Terminated model standalone evaluation')
log.info(f'AUC: {auc}')
