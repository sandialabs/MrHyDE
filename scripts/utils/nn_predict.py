#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:25:19 2022

@author: tmwilde
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    parser.add_argument('--model', default='subgrid_Model0.pt',
                        help='Saved model')
    parser.add_argument('--data', default='new_data.txt',
                        help='Saved model')
    parser.add_argument('--predictions', default='output_Model0.txt',
                        help='File with the output data')
    args = parser.parse_args()

    x_data = np.loadtxt(args.data,dtype=float)
    
    model = torch.load(args.model)
    
    x_data_torch = convert_tensor_2D(x_data)
    
    y_data = model(x_data_torch).float()
    y_data_np = convert_np_1D(y_data)
    
    y_data_np = np.abs(np.rint(y_data_np))
    y_data_int = y_data_np.astype(int)
    np.savetxt(args.predictions, y_data_int, fmt='%d') 

if __name__ == '__main__':
    main()