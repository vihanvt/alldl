import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
#load the dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['median_house_value'] = california_housing.target
features=["HouseAge","AveRooms","AveBedrms","Population","AveOccup","MedInc"]
inp_dim=len(features)
out_dim=1
num_layers=4
batch_size=16
x_train=df[features].values
y_train=df["median_house_value"].values
y_train=y_train.reshape(-1,1)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
y_train=scaler.fit_transform(y_train)
x_train=torch.from_numpy(x_train).float()
y_train=torch.from_numpy(y_train).float()
print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)
#for test data
x_test=df[features].values
y_test=df["median_house_value"].values
y_test=y_test.reshape(-1,1)
x_test=scaler.fit_transform(x_test)
y_test=scaler.fit_transform(y_test)
x_test=torch.from_numpy(x_test).float()
y_test=torch.from_numpy(y_test).float()
print(x_test)
print(y_test)
split_ratio=0.8
split_index=int(len(x_train)*split_ratio)
x_train,x_val=x_train[:split_index],x_train[split_index:]
y_train,y_val=y_train[:split_index],y_train[split_index:]
train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
test_dataset=torch.utils.data.TensorDataset(x_test,y_test)
val_dataset=torch.utils.data.TensorDataset(x_val,y_val)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size,shuffle=True)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=True)
#---- MODEL ARCH----"
class Model(nn.Module):
    def __init__(self,inp_dim,out_dim,num_layers,batch_size):
        super(Model,self).__init__()
        self.inp_dim=inp_dim
        self.out_dim=out_dim
        self.batch_size=batch_size
        self.num_layers=num_layers
        self.layer1=nn.Linear(inp_dim,32)
        self.layer2=nn.Linear(32,16)
        self.layer3=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        output=self.layer3(x)
        return output
epochs=100
lr=0.01
lossfunc=nn.MSELoss()
model=Model(inp_dim,out_dim,num_layers,batch_size)
optimizer=optim.Adam(model.parameters(),lr,weight_decay=1e-5)
for epoch in range(0,epochs):
    model.train()
    train_loss=0.0
    for inp,out in train_loader:
        optimizer.zero_grad()
        l1_lambda=1e-5
        l1_norm=sum(p.abs().sum() for p in model.parameters())
        output=model(inp)
        loss_val=lossfunc(output,out)
        loss_val=loss_val+l1_norm*l1_lambda
        loss_val.backward()
        optimizer.step()
        train_loss+=loss_val.item()
    model.eval()
    val_loss=0.0
    with torch.no_grad():
        for inp,out in val_loader:
            output=model(inp)
            loss_val=lossfunc(output,out)
            val_loss+=loss_val.item()
    test_loss=0.0
    for inp,out in test_loader:
        output=model(inp)
        loss_val=lossfunc(output,out)
        test_loss+=loss_val.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")