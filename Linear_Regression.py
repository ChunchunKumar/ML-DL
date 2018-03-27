import matplotlib.pyplot as plt
import numpy as np
import xlrd as x
file_location="E:/ML/Linear_Regression/Data.xlsx"
work_book=x.open_workbook(file_location)
sheet=work_book.sheet_by_index(0)
a=np.array([[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(sheet.nrows)])
data=a.T
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X2 = data.iloc[:,0:cols-1]
y2 = data.iloc[:,cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')