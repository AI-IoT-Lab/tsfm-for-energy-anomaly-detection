#!/usr/bin/env python
# coding: utf-8
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import optuna

import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import time
import joblib
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.")
warnings.filterwarnings('ignore',message='None of the inputs have requires_grad=True. Gradients will be None')
warnings.simplefilter(action='ignore', category=FutureWarning)
train_df=pd.read_csv("./DATASET/train.csv")
train_feats=pd.read_csv("./DATASET/train_features.csv")

for col in train_feats.columns:
    if train_feats[col].dtype=='object':
        print(col)
from momentfm.utils.anomaly_detection_metrics import adjbestf1

class TimeDataset(Dataset):
    def __init__(self,df):
        super().__init__()
        self.df=df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        return row.meter_reading,1,row.anomaly




DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
unique_ids=train_df['building_id'].unique()
df_list=[]
for b_id in unique_ids:

    building_df=train_df[train_df['building_id']==b_id].copy(deep=True)
    building_df['meter_reading']=building_df['meter_reading'].replace(float('nan'),
                                                                     building_df['meter_reading'].median())
    scaler=StandardScaler()
    building_df['meter_reading']=scaler.fit_transform(pd.DataFrame(building_df['meter_reading']))
    building_df.reset_index(drop=True,inplace=True)
    df_list.append(building_df)
print(len(unique_ids))
imputed_train=pd.concat(df_list)

class VarEncoderDecoder(nn.Module):
    def __init__(self,hidden_layers,hidden_size,latent_dim=32,seq_length=512):
        super().__init__()
        
        hidden_sizes=[]
        hidden_sizes.append(seq_length)
        for i in range(hidden_layers):
            hidden_sizes.append(hidden_size)
            hidden_size//=2
        
        self.encoder=nn.ModuleList()
        for i in range(1,len(hidden_sizes)):
            linear=nn.Linear(hidden_sizes[i-1],hidden_sizes[i])
            activation=nn.ReLU()
            norm=nn.BatchNorm1d(hidden_sizes[i])
            self.encoder.append(linear)
            self.encoder.append(norm)
            self.encoder.append(activation)

        self.encoder=nn.Sequential(*self.encoder)
        
        #mu,var
        self.enc_fc_mu=nn.Linear(hidden_sizes[-1],latent_dim)
        self.enc_fc_var=nn.Linear(hidden_sizes[-1],latent_dim)

        self.decoder_in=nn.Linear(latent_dim,hidden_sizes[-1])
        self.decoder=nn.ModuleList()
        hidden_sizes=list(reversed(hidden_sizes))
        for i in range(1,len(hidden_sizes)):
            linear=nn.Linear(hidden_sizes[i-1],hidden_sizes[i])
            activation=nn.LeakyReLU()
            norm=nn.BatchNorm1d(hidden_sizes[i])
            self.decoder.append(linear)
            self.decoder.append(norm)
            self.decoder.append(activation)        
        self.decoder=nn.Sequential(*self.decoder)
        
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self,X):
        X=self.encoder(X)
        # print(X.shape)
        mu=self.enc_fc_mu(X)
        log_var=self.enc_fc_var(X)
        X=self.reparameterize(mu,log_var)
        X=self.decoder_in(X)
        # print(X.shape)
        X=self.decoder(X)
        # print(X.shape)
        return X


def train_one_epoch(df_train,model,optimizer,loss_func,train_ids,
                   CONTEXT_LEN,BATCH_SIZE):
    model.train()
    
    losses=[]
    list_of_loaders=[]
    list_of_medians=[]
    for _id in train_ids:
        building=df_train[df_train['building_id']==_id].copy(deep=True)
        building.reset_index(drop=True,inplace=True)
        median=building['meter_reading'].median()
        dataset=TimeDataset(building)
        loader=DataLoader(dataset,batch_size=CONTEXT_LEN,shuffle=False,drop_last=True)
        iter_loader=iter(loader)
        list_of_loaders.append(iter_loader)
        list_of_medians.append(median)


    min_len=14
    for it in tqdm(range(min_len)):
        batch_x=[]
        batch_target=[]
        c=0
        encoder_hidden=None
        for ix,loader in enumerate(list_of_loaders):
            single_x,_,labels=next(loader)
            median=list_of_medians[ix]
            labels=labels.bool()
            target_x=single_x.clone().detach()
            target_x[labels]=median
            target_x=torch.unsqueeze(target_x,dim=0)
            batch_target.append(target_x)
            single_x=torch.unsqueeze(single_x,dim=0)
            batch_x.append(single_x)
            c+=1
            if (c!=len(list_of_loaders)) and (c % BATCH_SIZE != 0):
                continue
            batch_x=torch.cat(batch_x,dim=0).double().to(DEVICE)
            batch_target=torch.cat(batch_target,dim=0).double().to(DEVICE)
            output = model(batch_x) 
            loss = loss_func(output, batch_target)
            if math.isnan(loss.item()):
                print("Nan acquired! Quitting!")
                break
            losses.append(loss)
            batch_x=[]
            batch_target=[]
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
            optimizer.step()
    print(f"Training Loss: ",torch.mean(torch.tensor(losses)))
    return torch.mean(torch.tensor(losses))   



def test_model(df_val,model,val_ids):
    model.eval()
    all_scores=[]
    list_of_loaders=[]
    for b_id in val_ids:
        trues=[]
        preds=[]
        labels=[]
        first_building=df_val[df_val['building_id']==b_id].copy(deep=True)
        first_building.reset_index(drop=True,inplace=True)
        first_dataset=TimeDataset(first_building)
        loader=DataLoader(first_dataset, batch_size=512, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch_x, _,batch_labels in tqdm(loader, total=len(loader)):
                batch_x = batch_x.to(DEVICE).double()
                length=len(batch_x)
                if length<512:
                    zeros=torch.zeros(512-length)
                    zeros=zeros.to(DEVICE)
                    batch_x=torch.cat([batch_x,zeros],dim=0)
                    
                batch_x=torch.unsqueeze(batch_x,dim=0)
                batch_x=batch_x.to(DEVICE)
                output= model(batch_x) # [batch_size, window_size]
    
                trues.extend(batch_x.detach().squeeze().cpu().numpy()[:length])
                preds.extend(output.squeeze().cpu().numpy()[:length])
                labels.extend(batch_labels.detach().cpu().numpy())
            trues=np.array(trues)
            preds=np.array(preds)
            labels=np.array(labels)
            anomaly_scores=(trues-preds)**2
            score=adjbestf1(y_true=labels, y_scores=anomaly_scores)
            all_scores.append(score)
    return np.mean(all_scores)

def single_run(lr,epochs,beta_1,beta_2,hidden_size,num_layers,latent_dim):
    EPOCHS=epochs
    LR=lr
    TOTAL_LEN=365*24
    INPUT_SIZE=512
    CONTEXT_LEN=INPUT_SIZE
    HIDDEN_SIZE=hidden_size
    NUM_LAYERS=num_layers
    LATENT_DIM=latent_dim
    BATCH_SIZE=160
    scores=[]
    n_splits=5
    id_path="lead-val-ids/"
    for i in range(n_splits):
        val_ids=joblib.load(id_path+f"val_id_fold{i}.pkl")
        all_ids=imputed_train.building_id.unique()
        train_ids=[x for x in all_ids if x not in val_ids]
        print(f"FOLD: {i}")
        print("*"*20)
        model=VarEncoderDecoder(NUM_LAYERS,HIDDEN_SIZE,latent_dim=LATENT_DIM)
        model = model.to(DEVICE).double()
    
    
        df_train=imputed_train[~imputed_train.building_id.apply(lambda x: x in val_ids)]
        df_val=imputed_train[imputed_train.building_id.apply(lambda x: x  in val_ids)]
        loss_func=torch.nn.MSELoss()
        optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=(beta_1,beta_2))
        T_MAX=int((TOTAL_LEN/CONTEXT_LEN)/BATCH_SIZE*EPOCHS)
        val_score=0
        print(f"Starting f1_score={val_score}")
        print("TRAINING")
        print("*"*50)
        cur_score=0
        for epoch in range(EPOCHS):
            train_loss=train_one_epoch(df_train,model,optimizer,loss_func,train_ids,CONTEXT_LEN,BATCH_SIZE)
            val_score=test_model(df_val,model,val_ids)
            print(f"EPOCH= {epoch} Train Loss(MSE)={train_loss} Val Score(F1)= {val_score}")
            if val_score>cur_score:
                cur_score=val_score
                print(f'current best={cur_score}')
        
        scores.append(cur_score)
    return scores

def vaed_objective(trial):
    lr=trial.suggest_float('learning_rate', 1e-6, 1e-1)
    epochs=trial.suggest_int('epochs', 5,50)
    beta_1=trial.suggest_float('beta_1',0.1,1)
    beta_2=trial.suggest_float('beta_2',0.1,1)
    hidden_size=trial.suggest_categorical('hidden_size',[256,192,164,128,64])
    latent_dim=trial.suggest_categorical('latent_dim',[32,48,64,92])
    num_layers=trial.suggest_int('num_layers',1,1)
    scores=single_run(lr,epochs,beta_1,beta_2,hidden_size,num_layers,latent_dim)
    return np.mean(scores)

start_time = time.time()
study_vaed = optuna.create_study(direction='maximize')
study_vaed.optimize(vaed_objective, n_trials=50)
end_time = time.time()
elapsed_time_vaed = end_time - start_time
print(f"VAED training took {elapsed_time_vaed:.2f} seconds.")
best_params=study_vaed.best_trial.params
print(best_params)

joblib.dump(best_params,'best_variational_autoencoderdecoder_hyper_params.pkl')
joblib.dump(study_vaed,'vaed_study.pkl')


