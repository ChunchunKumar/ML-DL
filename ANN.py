import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
np.random.seed(1)
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
def load_planar_dataset():
    np.random.seed(1)
    m = 800  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y
def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
X, Y = load_planar_dataset()
X=np.asarray(X)
print(Y.shape)
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()
def initialize_parameters_deep(layers_dims):
    parameters={}
    L=len(layers_dims)-1
    for l in range(1,L+1):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*0.01
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))
    return parameters
def L_model_forward(X, parameters,layers_dims):
    caches={}
    A=X
    caches["A" + str(0)]=X
    L=len(layers_dims)-1
    for l in range(1,L):
        Z=np.dot(parameters["W"+str(l)],A)+parameters["b"+str(l)]
        caches["Z"+str(l)]=Z
        A = np.maximum(0, Z)
        caches["A"+str(l)]=A
        assert (Z.shape == (parameters["W" + str(l)].shape[0], A.shape[1]))
    ZL=np.dot(parameters["W"+str(L)],A)+parameters["b"+str(L)]
    caches["Z"+str(L)]=ZL
    AL=1/(1+np.exp(-ZL))
    assert(ZL.shape == (parameters["W"+str(L)].shape[0], AL.shape[1]))
    assert (ZL.shape == AL.shape)
    caches["A"+str(L)]=AL
    return AL, caches
def compute_cost(AL, Y):
    m=Y.shape[1]
    assert(AL.shape==Y.shape)
    cost=-(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    assert (cost.shape == ())
    return cost

def L_model_backward(AL, Y, caches, parameters,layers_dims):
    grads = {}
    L = len(layers_dims)-1
    m = AL.shape[1]
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    Z=caches["Z"+str(L)]
    s = 1 / (1 + np.exp(-Z))
    dZL = dAL * s * (1 - s)
    assert(s.shape==Z.shape)
    assert(dZL.shape==dAL.shape)
    dWL=(1/m)*np.dot(dZL,caches["A"+str(L-1)].T)
    dbL=(1/m)*np.sum(dZL,axis=1,keepdims=True)
    grads["dW"+str(L)]=dWL
    grads["db"+str(L)]=dbL
    dAl=np.dot(parameters["W"+str(L)].T,dZL)
    assert (dAl.shape == caches["A"+str(L-1)].shape)
    assert (dWL.shape == parameters["W"+str(L)].shape)
    assert (dbL.shape == parameters["b"+str(L)].shape)
    for l in reversed(range(1,L)):
       Z=caches["Z"+str(l)]
       assert(Z.shape==dAl.shape)
       dZl = np.array(dAl, copy=True)
       assert (dZl.shape==dAl.shape)
       dZl[Z <= 0] = 0
       assert (dZl.shape == Z.shape)
       dWl=(1/m)*np.dot(dZl,caches["A"+str(l-1)].T)
       dbl=(1/m)*np.sum(dZl,axis=1,keepdims=True)
       grads["dW"+str(l)]=dWl
       grads["db"+str(l)]=dbl
       dAl=np.dot(parameters["W"+str(l)].T,dZl)
       assert (dAl.shape == caches["A"+str(l-1)].shape)
       assert (dWl.shape == parameters["W"+str(l)].shape)
       assert (dbl.shape == parameters["b"+str(l)].shape)
    return grads
def update_parameters(parameters, grads, learning_rate, layers_dims):
    L=len(layers_dims)-1
    for l in range(1,L+1):
        assert(parameters["W"+str(l)].shape==grads["dW"+str(l)].shape)
        assert(parameters["b"+str(l)].shape==grads["db"+str(l)].shape)
        parameters["W"+str(l)]=parameters["W"+str(l)]-learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)]=parameters["b"+str(l)]-learning_rate*grads["db"+str(l)]
    return parameters
layers_dims = [2,30,10,1]
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters,layers_dims)
        cost = compute_cost(AL, Y)
        grads =L_model_backward(AL, Y, caches, parameters,layers_dims)
        parameters = update_parameters(parameters, grads, learning_rate, layers_dims)
        print("Cost after iteration %i: %f" % (i, cost))
        costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters
parameters = L_layer_model(X, Y, layers_dims, learning_rate=0.1,num_iterations =50000)


def predict(parameters, X):
       AL, cache = L_model_forward(X, parameters,layers_dims)
    predictions = (AL >= 0.5)
    return predictions
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()







