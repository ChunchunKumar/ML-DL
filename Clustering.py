import matplotlib.pyplot as plt
import numpy as np
import xlrd as x
from scipy.stats import norm
from random import randint
def model(X1,X2,mean,Var):
    n=mean.shape[0]
    mult=norm(mean[0][0],var[0][0]).pdf(X1)*norm(mean[1][0],var[1][0]).pdf(X2)
    return mult
file_location="E:/ML/Clustering/Data_KMean.xlsx"
work_book=x.open_workbook(file_location)
sheet=work_book.sheet_by_index(0)
a=np.array([[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(sheet.nrows)])
train=a.T
#plt.scatter(train[0,:],train[1,:])
def clustering(K,data):
        mean = np.random.uniform(3, 6, (data.shape[0],K))
        C = np.zeros((1, data.shape[1]))
        for k in range(20):
            for i in range(data.shape[1]):
                min_index = 0
                min = np.sum(np.square(data[:, i] - mean[:, 0]))
                for j in range(K):
                    if (np.sum(np.square(data[:, i] - mean[:, j])) < min):
                        min = np.sum(np.square(data[:, i] - mean[:, j]))
                        min_index = j
                C[0][i] = min_index
            # print(C)
            for i in range(K):
                sum = np.zeros((data.shape[0], 1))
                p = 0
                for j in range(data.shape[1]):
                    if (C[0][j] == i):
                        sum[:, 0] = sum[:, 0] + data[:, j]
                        p = p + 1
                if (p != 0):
                    mean[:, i] = sum[:, 0] / (p)
        return C, mean
Error_K=np.zeros((1,10))
K=2
for u in range(10):
    error_final = 10000
    C_final = np.zeros((1, train.shape[1]))
    mean_final = np.random.uniform(0, 20, (train.shape[0], K))
    for i in range(20):
        C, mean = clustering(K, train)
        error = 0
        for i in range(train.shape[1]):
                error = error+np.sum(np.square(train[:, i] - mean[:, int(C[0][i])]))
        error = error / train.shape[1]
        if (error_final > error):
            C_final = C
            error_final = error
            mean_final = mean
    Error_K[0][u]=error_final
    print(C_final)
    print(error_final)
    plot=np.zeros(train.shape)
    for j in range(K):
        plot = np.zeros(train.shape)
        for l in range(train.shape[1]):
            if (C[0][l] == j):
                plot[:, l] = train[:, l]
        plt.scatter(plot[0, :], plot[1, :], marker='*')
    plt.scatter(mean[0, :], mean[1, :])
    plt.title(K)
    plt.show()

    '''plot = np.zeros(train.shape)
    for j in range(K):
        plot = np.zeros(train.shape)
        for l in range(train.shape[1]):
            if (C_final[0][l] == j):
                plot[:, l] = train[:, l]
        plt.scatter(plot[0, :], plot[1, :], marker='*')
    plt.scatter(mean_final[0, :], mean_final[1, :])
    plt.show() '''
    K=K+1
plt.scatter(np.linspace(2,11,10),Error_K)
plt.show()

