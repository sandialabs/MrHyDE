#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 07:18:39 2022

@author: tmwilde
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,1)
        self.smax = nn.Softmax(dim=0)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = x
        return output

def initialize_weights(m):

  if isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.uniform_(m.bias.data, 0.0, 1.0)
      
def convert_tensor_1D(X):
    X_tensor = torch.zeros(X.shape[0],1)
    for i in range(X.shape[0]) :
        X_tensor[i,0] = X[i]
    return X_tensor

def convert_tensor_2D(X):
    X_tensor = torch.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            X_tensor[i,j] = X[i,j]
    return X_tensor

def convert_np_1D(X):
    X_np = np.zeros(X.shape[0])
    for i in range(X.shape[0]) :
        X_np[i] = X[i,0]
    return X_np

def convert_np_2D(X):
    X_np = np.zeros(X.shape[0],X.shape[1])
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            X_np[i][j] = X[i,j]
    return X_np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Binary Classifier Example')
    parser.add_argument('--lr', type=float, default=1.0e-2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--nn-model', default='nn_model.pt',
                        help='pytorch model')
    parser.add_argument('--input-data', default='Model0_0_input.txt',
                        help='File with the input data')
    parser.add_argument('--output-data', default='Model0_0_output.txt',
                        help='File with the output data')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    device = "cpu"

    x_data = np.loadtxt(args.input_data,dtype=float)
    y_data = np.loadtxt(args.output_data,dtype=int)

    Ntrain = int(np.round(0.8*x_data.shape[0]))

    x_train = x_data[:Ntrain,:]
    x_test = x_data[Ntrain:,:]

    y_train = y_data[:Ntrain]
    y_test = y_data[Ntrain:]
    
    x_train_torch = convert_tensor_2D(x_train)
    y_train_torch = convert_tensor_1D(y_train)
    x_test_torch = convert_tensor_2D(x_test)
    y_test_torch = convert_tensor_1D(y_test)

    model = Net().to(device)
    model.apply(initialize_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # I like to manually adapt the learning rate as it approaches a local min
    adapt_lr = True
    lr_iter = 0
    adapt_patience = 100
    adapt_lookback = 10
    
    Nopt = int(10e3)
    train_errs = np.zeros(Nopt)
    test_errs = np.zeros(Nopt)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    #loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')    
    for t in range(Nopt):
        optimizer.zero_grad()
        output = model(x_train_torch).float()
        #print(output.shape)

        loss = loss_fn(output, y_train_torch)
        
        train_errs[t] = loss.item()
        #print(loss.item())
        
        
        y_test_pred = model(x_test_torch).float()
        #testloss = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(y_test_pred, y_test_torch)
        test_errs[t] = test_loss.item()
        
        lr_iter += 1
        
        if adapt_lr :
            if lr_iter>adapt_patience and train_errs[t]>1.0*train_errs[t-adapt_lookback] :
                optimizer.param_groups[0]['lr'] *= 0.5
                lr_iter = 0 # reset to 0 for new lr
                print('New learning rate: ' + str(optimizer.param_groups[0]['lr']))
        
        loss.backward()
        optimizer.step()
        if t % 1000 == 0:
            print("Finished optimization iteration "+str(t) + " with loss: " + str(train_errs[t]))
    
    
    y_test_surr = model(x_test_torch).float()
    y_test_surr_np = convert_np_1D(y_test_surr)
    
    
    error = abs(np.rint(y_test)-np.rint(y_test_surr_np))
    unique, counts = np.unique(error, return_counts=True)
    print("Testing accuracy: " + str(counts[0]/y_test.shape[0]*100) + "%")
    
    if args.save_model:
#        torch.save(model.state_dict(), "subgrid_Model0.pt")
        torch.save(model, args.nn_model)


if __name__ == '__main__':
    main()