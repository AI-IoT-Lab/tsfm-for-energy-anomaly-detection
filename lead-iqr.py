# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import optuna
import time
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_df = pd.read_csv("./train.csv")

train_df

unique_ids = train_df.building_id.unique()

df_list = []

for b_id in unique_ids:
    building_df = train_df.groupby('building_id').get_group(b_id).copy(deep=True)
    building_df['meter_reading'] = building_df['meter_reading'].replace(float('nan'),
                                                                        building_df['meter_reading'].median())
    building_df.reset_index(drop=True, inplace=True)
    df_list.append(building_df)

print(len(df_list))

imputed_train = pd.concat(df_list)

imputed_train

imputed_train.anomaly.value_counts()

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

seed = 42
import joblib

def calculate_quartiles_and_iqr(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    # Calculate IQR
    iqr = q3 - q1

    return q1, q3, iqr



f1_scores = []
prec_scores = []
recall_scores = []
all_ids = imputed_train.building_id.unique()
for _id in all_ids:
    building = imputed_train[imputed_train['building_id'] == _id].copy(deep=True)
    X_train = building['meter_reading']
    y_train = building['anomaly'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(pd.DataFrame(X_train))
    ind_best_f1 = 0
    q1, q3, iqr = calculate_quartiles_and_iqr(X_train)
    for K in range(5, 45, 5):
        K /= 10  # 0.5,1,1.5,2,...
        low = q1 - K * iqr
        high = q3 + K * iqr
        y_pred = np.zeros(len(y_train))
        bool_array = (X_train > high) | (X_train < low)
        bool_array = bool_array.reshape(-1)
        y_pred[bool_array] = 1

        score_f1 = f1_score(y_train, y_pred)

        if score_f1 > ind_best_f1:
            score_prec = precision_score(y_train, y_pred)
            score_recall = recall_score(y_train, y_pred)
            ind_best_f1 = score_f1
            ind_best_prec = score_prec
            ind_best_recall = score_recall
    f1_scores.append(ind_best_f1)
    prec_scores.append(ind_best_prec)
    recall_scores.append(ind_best_recall)

print(f1_scores)

print(prec_scores)

print(recall_scores)

print(np.mean(f1_scores))

print(np.mean(prec_scores))

print(np.mean(recall_scores))