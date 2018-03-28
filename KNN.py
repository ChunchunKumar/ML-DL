import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
data = pd.read_csv("C:/Users/Manish/Desktop/Deep Learning/data.csv")
train=np.array(data)
data = pd.read_csv("C:/Users/Manish/Desktop/Deep Learning/test.csv")
test=np.array(data)
def KNN(Data,test, K):
    dist=[]
    test=np.array(test)
    #print(Data[0,:])
    #print(test)
    #print(Data[0,0:Data.shape[1]-1]-test)
    for i in range(Data.shape[0]):
        dist.append(np.sum(np.square(Data[i,0:Data.shape[1]-1]-test)))
    sort=sorted(dist)
    K_neighbour=[]
    for i in range(K):
        K_neighbour.append(Data[dist.index(sort[i]),Data.shape[1]-1])
        #print(dist.index(sort[i]))
    K_neighbour=np.array(K_neighbour)
    max=np.max(np.unique(K_neighbour,return_counts=True)[1])
    List_Neighbour=list(np.unique(K_neighbour,return_counts=True)[0])
    Frequency=list(np.unique(K_neighbour,return_counts=True)[1])
    predict=List_Neighbour[Frequency.index(max)]
    return(predict)
Test_error=[]
Train_error=[]
print(train[train.shape[0]-1,4]==KNN(train,train[train.shape[0]-1,0:4],1))
K=1
for j in range(35):
    train_error=0
    for i in range(train.shape[0]):
        if(train[i,4]==KNN(train,train[i,0:4],K)):
            train_error=train_error+1
    #print((train_error/train.shape[0])*100)
    Train_error.append(100-(train_error/train.shape[0])*100)
    test_error=0
    for i in range(test.shape[0]):
        if(test[i,4]==KNN(train,test[i,0:4],K)):
            test_error=test_error+1
    print(100-(test_error/test.shape[0]) * 100)
    Test_error.append(100-(test_error/test.shape[0])*100)
    K=K+1
plt.scatter(range(35),Train_error)
plt.show()
plt.scatter(range(35),Test_error)
plt.show()
from sklearn.neighbors import KNeighborsClassifier
K=1
Train_error=[]
Test_error=[]
for j in range(35):
    test_error=0
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(train[:, 0:4], train[:, 4])
    for i in range(test.shape[0]):
        if(test[i,4]==neigh.predict([test[i,0:4]])[0]):
            test_error=test_error+1
    #print(100-(test_error/test.shape[0])*100)
    Test_error.append(100-(test_error/test.shape[0])*100)
    train_error = 0
    for i in range(train.shape[0]):
        if (train[i, 4] == neigh.predict([train[i, 0:4]])[0]):
            train_error = train_error + 1
    # print((train_error/train.shape[0])*100)
    Train_error.append(100 - (train_error / train.shape[0]) * 100)
    K=K+1
plt.scatter(range(35),Train_error)
plt.show()
plt.scatter(range(35),Test_error)
plt.show()
#print(neigh.predict([[5.1,3.5,1.4,0.2]])[0])



