#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:26:37 2018

@author: elenaxu
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
boston = datasets.load_boston()
data = boston.data
target = boston.target
#print boston.data.shape[0] 
#print boston.shape      (506,13)
#datashape[0]=506

#normalize data, axis=0-->compare by columns  
max_data=np.amax(data, axis=0)
min_data=np.amin(data, axis=0)
max_target=np.amax(target)
min_target=np.amin(target)
#there is only one list of numbers in target 
#print max_data, min_data, max_target

for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            data[i][j]= (data[i][j]-min_data[j]) / (max_data[j]-min_data[j])
for i in range(0,len(target)):
    target[i]= (target[i]-min_target) / (max_target-min_target)
    

#insert one more variable x0=1 at each array, position 0, value=1
data = np.insert(boston.data,0,1,axis=1)


#easier to set a regular list first and return to nplist later
trainingX= []
trainingY= []

testX= []
testY= []

for i in range(0,data.shape[0]):
#for i in the 506 datalist 
    if (i % 10 == 0):
    #if the current index/10 have a reminder of 0
        testX.append(data[i])
        testY.append(target[i])
    else:
        trainingX.append(data[i])
        trainingY.append(target[i])

#switch the regular list into a nplist
trainingX=np.array(trainingX)
trainingY=np.array(trainingY)
testX=np.array(testX)
testY=np.array(testY)

#print trainingX.shape   testX。shape(51, 14)

def model(b,x):
    y=0
    for i in range(0,data.shape[1]):
        y += b[i]*x[i]
    return y

x=trainingX
y=trainingY
b=[0]*(trainingX.shape[1])
#append 0.0 in b for 14 times



def rmse(predictions, target):
    return np.sqrt(((predictions - target) ** 2).mean())
#do not need for loop for each index--predictions and target are both lists with same size


learning_rate= [0.00001,0.0001,0.001,0.01,0.1,1]
for a in learning_rate:
    #learning_rate=0.01
    max_epochs = 10
    current_epoch=0
    
    rmse_list=[]
    epoch=[]
    while current_epoch < max_epochs:
        predictions=[]
        for i in range(0,trainingX.shape[0]):    
            error = model(b,x[i]) - y[i]
            for j in range(0,trainingX.shape[1]):
                b[j] = b[j] - a*error*x[i][j]
            
            prediction = model(b,x[i])
            predictions.append(prediction)
        #get b first and then do predictions
        
        rmses=rmse(predictions, trainingY)
        rmse_list.append(rmses)
    
        current_epoch += 1
        epoch.append(current_epoch)
    print "Learning rate is", a
    print "RMS Error is", rmse_list

#plot epoch vs RMSE 
    plt.plot (epoch, rmse_list)
    plt.xlabel('epoch')
    plt.title('Epoch vs RMSE')
    plt.ylabel('Error')
    plt.show() 

    
    