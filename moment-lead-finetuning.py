#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import optuna
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.")
warnings.filterwarnings('ignore',message='None of the inputs have requires_grad=True. Gradients will be None')
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch
import os
import matplotlib.pyplot as plt
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("./DATASET/train.csv")
train_feats=pd.read_csv("./DATASET/train_features.csv")
train_feats.info(verbose=True)

from sklearn.preprocessing import LabelEncoder
for col in train_feats.columns:
    if train_feats[col].dtype=='object':
        print(col)



from momentfm import MOMENTPipeline

from momentfm.utils.anomaly_detection_metrics import adjbestf1
from torch.utils.data import Dataset, DataLoader

class TimeDataset(Dataset):
    def __init__(self,df):
        super().__init__()
        self.df=df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        return row.meter_reading,1,row.anomaly


# In[15]:
DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from tqdm import tqdm
trues, preds, labels = [], [], []

unique_ids=train_df['building_id'].unique()
b_id=unique_ids[8]
print(b_id)

building_df=train_df[train_df['building_id']==b_id].copy(deep=True)
building_df['meter_reading']=building_df['meter_reading'].replace(float('nan'),
                                                                 building_df['meter_reading'].median())
building_df.anomaly.value_counts()

building_df.loc[building_df.anomaly==1,'meter_reading']=np.nan
building_df.meter_reading.isnull().sum()
building_df.meter_reading.ffill(inplace=True)
building_df.meter_reading.isnull().sum()
df_list=[]
for b_id in unique_ids:

    building_df=train_df[train_df['building_id']==b_id].copy(deep=True)
    building_df['meter_reading']=building_df['meter_reading'].replace(float('nan'),
                                                                     building_df['meter_reading'].median())

    building_df.reset_index(drop=True,inplace=True)
    df_list.append(building_df)
print(len(unique_ids))
imputed_train=pd.concat(df_list)
imputed_train

class Masking:
    def __init__(self, mask_ratio=0.3):
        self.mask_ratio = mask_ratio
    
    def generate_mask(self, x, input_mask=None):
        batch_size = x.shape[0]
        window_size = x.shape[2]
        num_masks = int(window_size * self.mask_ratio)
        
        mask = torch.ones((batch_size,window_size), dtype=torch.bool)
        for i in range(batch_size):
            mask_indices = torch.randperm(window_size)[:num_masks]
            mask[i,mask_indices] = False
        return mask

def train_one_epoch(df_train,model,optimizer,scheduler,loss_func,train_ids,
                   mask_generator,CONTEXT_LEN,BATCH_SIZE):
    model.train()
    
    losses=[]
    for _id in train_ids:
        building=df_train[df_train['building_id']==_id].copy(deep=True)
        building.reset_index(drop=True,inplace=True)
        median=building['meter_reading'].median()
        dataset=TimeDataset(building)
        loader=DataLoader(dataset,batch_size=CONTEXT_LEN,shuffle=True,drop_last=True)
        batch_x=[]
        batch_masks=[]
        batch_target=[]
        c=0
        for single_x,single_mask,labels in tqdm(loader,total=len(loader)):
            labels=labels.bool()
            target_x=single_x.clone().detach()
            target_x[labels]=median

            target_x=torch.unsqueeze(target_x,dim=0)
            target_x=torch.unsqueeze(target_x,dim=0)
            batch_target.append(target_x)
            
            single_x=torch.unsqueeze(single_x,dim=0)
            single_mask=torch.unsqueeze(single_mask,dim=0)
            single_x=torch.unsqueeze(single_x,dim=0)
            # single_mask=torch.unsqueeze(single_mask,dim=0)
            batch_x.append(single_x)
            batch_masks.append(single_mask)
            c+=1
            if (c!=len(loader)) and (c % BATCH_SIZE != 0):
                continue
            batch_x=torch.cat(batch_x,dim=0).float().to(DEVICE)
            batch_target=torch.cat(batch_target,dim=0).float().to(DEVICE)
            batch_masks=torch.cat(batch_masks,dim=0).float().to(DEVICE)
            mask = mask_generator.generate_mask(
                x=batch_x, input_mask=batch_masks).to(DEVICE).bool()
        
            # Forward
            # print(batch_x.shape,batch_masks.shape,mask.shape)
            n_channels = batch_x.shape[1]
            # print(n_channels)
            # mask = mask.unsqueeze(1).repeat(1, 1, 1).bool()
            # print(mask.shape)
            output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask) 
            
            loss = loss_func(output.reconstruction, batch_target)
            if math.isnan(loss.item()):
                print("Nan acquired! Quitting!")
                break
            # print(f"loss: {loss.item()}")
            losses.append(loss)
            batch_x=[]
            batch_masks=[]
            batch_target=[]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
            optimizer.step()
    return torch.mean(torch.tensor(losses))   

def test_model(df_val,model,val_ids):
    model.eval()
    all_scores=[]
    for b_id in val_ids:
        trues=[]
        preds=[]
        labels=[]
        first_building=df_val[df_val['building_id']==b_id].copy(deep=True)
        first_building.reset_index(drop=True,inplace=True)
        # first_building['meter_reading']=first_building['meter_reading'].replace(float('nan'),first_building['meter_reading'].median())
        first_dataset=TimeDataset(first_building)
        loader=DataLoader(first_dataset, batch_size=512, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch_x, _,batch_labels in tqdm(loader, total=len(loader)):
                batch_x = batch_x.to(DEVICE).float()
                length=len(batch_x)
                batch_masks=torch.ones(length)
                if length<512:
                    zeros=torch.zeros(512-length)
                    zeros=zeros.to(DEVICE)
                    batch_x=torch.cat([batch_x,zeros],dim=0)
                    padded_mask=torch.zeros(512-length)
                    # padded_mask=padded_mask.to(DEVICE)
                    
                    batch_masks=torch.cat([batch_masks,padded_mask],dim=0)
                batch_x=torch.unsqueeze(batch_x,dim=0)
                batch_x=torch.unsqueeze(batch_x,dim=0)
                batch_masks=torch.unsqueeze(batch_masks,dim=0)
                batch_masks = batch_masks.to(DEVICE)
                batch_x=batch_x.to(DEVICE)
                output = model(x_enc=batch_x,input_mask=batch_masks) # [batch_size, n_channels, window_size]
    
                trues.extend(batch_x.detach().squeeze().cpu().numpy()[:length])
                preds.extend(output.reconstruction.detach().squeeze().cpu().numpy()[:length])
                labels.extend(batch_labels.detach().cpu().numpy())
            trues=np.array(trues)
            preds=np.array(preds)
            labels=np.array(labels)
            anomaly_scores=(trues-preds)**2
            score=adjbestf1(y_true=labels, y_scores=anomaly_scores)
            all_scores.append(score)
    return np.mean(all_scores)

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import joblib

def single_run(lr,epochs,mask_size,beta_1,beta_2):
    EPOCHS=epochs
    LR=lr
    TOTAL_LEN=365*24
    CONTEXT_LEN=512
    BATCH_SIZE=8
    mask=Masking(mask_size)
    scores=[]
    n_splits=5
    id_path="/BASU_LEAD/validation_folds/"
    for i in range(n_splits):
        val_ids=joblib.load(id_path+f"val_id_fold{i}.pkl")
        all_ids=imputed_train.building_id.unique()
        train_ids=[x for x in all_ids if x not in val_ids]
        print(f"FOLD: {i}")
        print("*"*20)
        model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={"task_name": "reconstruction"},  # For anomaly detection, we will load MOMENT in `reconstruction` mode
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )
        model.init()
        model = model.to(DEVICE).float()
    
    
        
        df_train=imputed_train[~imputed_train.building_id.apply(lambda x: x in val_ids)]
        df_val=imputed_train[imputed_train.building_id.apply(lambda x: x  in val_ids)]
        loss_func=torch.nn.MSELoss()
        optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=(beta_1,beta_2))
        T_MAX=int((TOTAL_LEN/CONTEXT_LEN)/BATCH_SIZE*EPOCHS)
        scheduler=CosineAnnealingLR(optimizer,T_max=T_MAX,eta_min=1e-7)
        val_score=0
        print("TRAINING")
        print("*"*50)
        cur_score=0
        for epoch in range(EPOCHS):
            train_loss=train_one_epoch(df_train,model,optimizer,scheduler,loss_func,train_ids,mask,CONTEXT_LEN,BATCH_SIZE)
            val_score=test_model(df_val,model,val_ids)
            print(f"EPOCH= {epoch} Train Loss(MSE)={train_loss} Val Score(F1)= {val_score}")
            if val_score>cur_score:
                cur_score=val_score
                # torch.save(model.state_dict(),f'model_finetuned_{i}.pkl')
                print(f'current best={cur_score}')
        
        scores.append(cur_score)
    return scores



def moment_objective(trial):
    lr=trial.suggest_float('learning_rate', 1e-6, 1e-1)
    epochs=trial.suggest_int('epochs', 1,25)
    mask=trial.suggest_float('mask_size',0.1,0.5)
    beta_1=trial.suggest_float('beta_1',0.5,1.0)
    beta_2=trial.suggest_float('beta_2',0.5,1.0)
    scores=single_run(lr,epochs,mask,beta_1,beta_2)
    return np.mean(scores)


import time

start_time = time.time()
study_moment = optuna.create_study(direction='maximize')
study_moment.optimize(moment_objective, n_trials=50)
end_time = time.time()
elapsed_time_cb = end_time - start_time
print(f"Moment tuning took {elapsed_time_cb:.2f} seconds.")



best_params=study_moment.best_trial.params

print(best_params)
joblib.dump(best_params,'best_params_moment_50.pkl')

