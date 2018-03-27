import matplotlib.pyplot as plt
import numpy as np
import xlrd as x
from scipy.stats import norm
def model(X1,X2,mean,Var):
    n=mean.shape[0]
    mult=norm(mean[0][0],var[0][0]).pdf(X1)*norm(mean[1][0],var[1][0]).pdf(X2)
    return mult
file_location="E:/ML/Anomaly and recommmender system/Data1.xlsx"
work_book=x.open_workbook(file_location)
sheet=work_book.sheet_by_index(0)
a=np.array([[sheet.cell_value(r,c) for c in range(sheet.ncols-1)] for r in range(sheet.nrows)])
train=a.T
file_location="E:/ML/Anomaly and recommmender system/Data2.xlsx"
work_book=x.open_workbook(file_location)
sheet=work_book.sheet_by_index(0)
a=np.array([[sheet.cell_value(r,c) for c in range(sheet.ncols-1)] for r in range(sheet.nrows)])
test=a.T
#plt.hist(train[0,:])
#plt.show()
mean=np.array([np.sum(train,axis=1)/train.shape[1]]).T
var=np.array([np.sum(np.square(train-mean),axis=1)/train.shape[1]]).T
x= np.linspace(5, 25, 3000)
x1,x2=np.meshgrid(x,x)
cp=plt.contour(x1,x2,norm(mean[0][0],var[0][0]).pdf(x1)*norm(mean[1][0],var[1][0]).pdf(x2),70)
plt.colorbar()
plt.scatter(train[0,:],train[1,:])
plt.scatter(test[0,:],test[1,:],marker='*')
plt.title("Anomaly Detection")
plt.show()