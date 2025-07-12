import numpy as np
import copy

def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    
    return params



def initialize_parameters_deep(layer_dims):

    
    parameters = {}
    
    L= len(layer_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def initialize_parameters_deep_xavier(layer_dims):

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def initialize_parameters_he(layers_dims):
    np.random.seed(3)

    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) *np.sqrt(2./layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters




def linear_forward(A, W, b):

    Z = np.dot(W, A) +b

    cache = (A, W, b)
    
    return Z, cache


def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    A = 1 / (1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache



def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    
    return A, cache


def L_model_forward(X, parameters):

    A = X
    L = len(parameters)//2
    caches = []

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-8, 1 - 1e-8)
    j = -(1/m) * np.sum((Y*np.log(AL)) + ((1-Y)*np.log(1-AL)))
    j = np.squeeze(j)
    
    return j


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def sigmoid_backwards(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backwards(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == "sigmoid":
        dZ = sigmoid_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
    
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    # Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    
    for l in range(L-2, -1, -1):
        dA_prev = dA_prev_temp
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp        
        
    
    return grads
    
def L_model_backward_direct(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dZL = (AL-Y)

    current_cache = caches[-1]
    linear_cache, activation_cache = current_cache

    A_prev, W, b = linear_cache
    dA_prev_temp = np.dot(W.T, dZL)
    dW_temp = (1/m) * np.dot(dZL, A_prev.T)
    db_temp = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    
    for l in range(L-2, -1, -1):
        dA_prev = dA_prev_temp
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp        
        
    
    return grads


def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(params)//2
    
    for l in range(1, L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * grads["dW"+str(l)])
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * grads["db"+str(l)])

    return parameters
    

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def compute_cost_with_regularization(al, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2
    cross_entropy_cost = compute_cost(al, Y)
    L2_regularization_cost = 0

    for l in range(1, L+1):
        L2_regularization_cost += np.sum(np.square(parameters["W"+str(l)]))

    L2_regularization_cost = (1./m)*(lambd/2)*L2_regularization_cost
    
    cost = cross_entropy_cost + L2_regularization_cost
    return cost




def linear_backward_reg(dZ, cache, lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T) + ((lambd/m)*W)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward_reg(dA, cache, lambd=0):
    linear_cache, activation_cache = cache

    dZ = relu_backwards(dA, activation_cache)
    # dZ = np.multiply(dA, np.int64(linear_cache[0] > 0))
    dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambd)
    # print(activation_cache.shape)
    return dA_prev, dW, db

def L_model_backward_direct_reg(X, Y, caches, lambd, al):
    grads = {}
    L = len(caches)
    m = X.shape[1]
    # Y = Y.reshape(AL.shape)

    current_cacheL = caches[-1]
    linear_cache, activation_cache = current_cacheL
    # print(activation_cache.shape)
    
    dZL = (al-Y)

    A_minus_l, WL, bL = linear_cache
    

    dW_temp = 1/m * np.dot(dZL, A_minus_l.T) + ((lambd/m)*WL)
    db_temp = 1/m * np.sum(dZL, axis=1, keepdims=True)
    dA_prev_temp = np.dot(WL.T, dZL)
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp 
    grads["db" + str(L)] = db_temp
    
    
    for l in reversed(range(L-1)):
        dA_prev = dA_prev_temp
        (Al_minus_1, Wl, Bl), dZl = caches[l] 
        # print(Al.shape)
        dA_prev_temp, dW_plus_1, db_plus_1 = linear_activation_backward_reg(dA_prev, ((Al_minus_1, Wl, Bl), dZl), lambd)

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_plus_1
        grads["db" + str(l+1)] = db_plus_1
        
    return grads



def linear_activation_forward_dropout(A_prev, W, b, activation, keep_prob=1):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        # D = np.random.rand(*A.shape)
        # mask = (D < keep_prob).astype(int)
        # A = (A*mask)/keep_prob
        mask = None
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        D = np.random.rand(*A.shape)
        mask = (D < keep_prob).astype(int)
        A = (A*mask)/keep_prob
    cache = (linear_cache, activation_cache, mask)
    
    return A, cache

def forward_propagation_with_dropout(X, parameters, keep_prob = 1):

    np.random.seed(1)
    L = len(parameters)//2
    caches = []
    A = X

    
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward_dropout(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu", keep_prob)
        caches.append(cache)

    AL, cache = linear_activation_forward_dropout(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid", keep_prob)
    caches.append(cache)
    
    return AL, caches
    
def linear_activation_backward_dropout(dA, cache, activation, keep_prob=1):
    linear_cache, activation_cache, mask = cache

    if activation == "relu":
        dZ = relu_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        dA_prev = (dA_prev * mask) / keep_prob
    if activation == "sigmoid":
        dZ = sigmoid_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
    
def backward_propagation_with_dropout(X, Y, caches, keep_prob, AL):
    grads = {}
    L = len(caches)
    m = X.shape[1]
    Y = Y.reshape(AL.shape)
    
    dZL = (AL-Y)
    current_cacheL = caches[-1]
    linear_cache, activation_cache, D3 = current_cacheL
    _, _, D2 = caches[-2]
    A_prev_L, WL, bL = linear_cache

    dA_prev_temp = np.dot(WL.T, dZL)
    dW_temp = (1/m) * np.dot(dZL, A_prev_L.T)
    db_temp = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    dA_prev_temp = (dA_prev_temp*D2)/keep_prob
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp 
    grads["db" + str(L)] = db_temp

    
    for l in reversed(range(L-1)):
        dA_prev = dA_prev_temp
        linear_cache, activation_cache, mask  = caches[l]
        _,_,Dl  = caches[l-1]
        # print(activation_cache.shape)
        if l > 0:
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward_dropout(dA_prev, (linear_cache, activation_cache, Dl), "relu", keep_prob)
        else:
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, (linear_cache, activation_cache), "relu")
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        
    return grads
