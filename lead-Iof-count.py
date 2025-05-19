# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

train_df=pd.read_csv("./train.csv")

train_df

unique_ids=train_df.building_id.unique()

df_list=[]

for b_id in unique_ids:
    building_df=train_df.groupby('building_id').get_group(b_id).copy(deep=True)
    building_df['meter_reading']=building_df['meter_reading'].replace(float('nan'),
                                                    building_df['meter_reading'].median())
    building_df.reset_index(drop=True,inplace=True)
    df_list.append(building_df)

print(len(df_list))

imputed_train=pd.concat(df_list)

imputed_train

imputed_train.anomaly.value_counts()

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

seed=42
import joblib

best_params={'n_neighbors': 500,
             'algorithm': 'brute',
             'leaf_size': 458,
             'metric': 'minkowski',
             'p': 48,
             'contamination': 0.019012254771883536,
             'novelty': False}



f1_scores = []
prec_scores=[]
recall_scores=[]
all_ids=imputed_train.building_id.unique()
fold_scores=[]
for _id in all_ids:
    building = imputed_train[imputed_train['building_id']==_id].copy(deep=True)
    X_train=building['meter_reading']
    y_train = building['anomaly']
    scaler=StandardScaler()
    X_train=scaler.fit_transform(pd.DataFrame(X_train))

    ind_best_f1=0
    for cont in range(1,87,5):
        cont/=1000
        best_params['contamination']=cont
        model = LocalOutlierFactor( **best_params)
        y_pred=model.fit_predict(X_train)
        y_pred=np.where(y_pred==-1,1,0)

        score_f1=f1_score(y_train,y_pred)

        if score_f1> ind_best_f1:
            score_prec=precision_score(y_train,y_pred)
            score_recall=recall_score(y_train,y_pred)
            ind_best_f1=score_f1
            ind_best_prec=score_prec
            ind_best_recall=score_recall
    f1_scores.append(ind_best_f1)
    prec_scores.append(ind_best_prec)
    recall_scores.append(ind_best_recall)


print(f1_scores)

print(prec_scores)

print(recall_scores)

print(np.mean(f1_scores))

print(np.mean(prec_scores))

print(np.mean(recall_scores))