import matplotlib.pyplot as mt
import os
import numpy as np
import csv
rows = []
filename="ex2data2.csv"
Y_train=np.array([data[:,2]])
def sigmoid(Z=[0,0]):
    h=1/(1+np.exp(-1*Z))
    return h
def cost(X_train,Y_train,W,B):
    J=0
    Z=np.dot(W,X_train)+B
    J=-(1/(Y_train.shape[1]))*(np.dot(Y_train,np.log(sigmoid(Z)).T)+np.dot(1-Y_train,np.log(1-sigmoid(Z)).T))
    return J
def gradient(X_train,Y_train,W,B):
    dW=[]
    Z=np.dot(W,X_train)+B
    for i in np.arange(X_train.shape[0]+1):
        if i==0:
            dB=(1/Y_train.shape[1])*np.sum(sigmoid(Z)-Y_train)
        else:
            dW.append((1/Y_train.shape[1])*np.dot(sigmoid(Z)-Y_train,X_train[i-1,:].T))
    return dW,dB
def logistic_regression(X_train,Y_train):
    W=np.array([[0.0001,0.0001]])
    B=0.001
    n_iteration=100000
    J=[]
    for i in np.arange(n_iteration):
       J.append(cost(X_train,Y_train,W,B))
       dW,dB=gradient(X_train,Y_train,W,B)
       W=W-0.001*np.array(dW).T
       B=B-0.001*dB
    return W,B,J
'''mW,mB,J=logistic_regression(X_train,Y_train)
l=-1*(mB+mW[0,0]*X_train[0,:])/mW[0,1]
mt.subplot(131)
mt.plot(np.array(J)[:,:,0])
mt.title("CostvSIteration")
mt.ylabel("Cost")
mt.xlabel("iteration")
mt.subplot(132)
mt.scatter(data[:,0],data[:,1],c=data[:,2],)
mt.scatter(data[:,0],l,marker='*')
mt.xlabel("First feature ")
mt.show()
print(sigmoid(np.dot(mW,X_train)+mB))


#print(np.array(X_train).shape)'''


