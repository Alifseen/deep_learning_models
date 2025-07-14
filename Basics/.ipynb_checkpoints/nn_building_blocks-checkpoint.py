import numpy as np
import copy
import math

# def initialize_parameters(n_x, n_h, n_y):
    
#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros((n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros((n_y, 1))

#     params = {
#         "W1":W1,
#         "b1":b1,
#         "W2":W2,
#         "b2":b2
#     }

    
#     return params



def initialize_parameters(layer_dims):

    np.random.seed(3)
    
    parameters = {}
    
    L= len(layer_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def initialize_parameters_xavier(layer_dims):

    # np.random.seed(1)
    
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def initialize_parameters_he(layers_dims):
    
    # np.random.seed(3)

    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) *np.sqrt(2./layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters



##############################################################################################################################



def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) +b
    cache = (A_prev, W, b)
    return Z, cache


def sigmoid(Z):
    A = 1 / (1+np.exp(-Z))
    Z_cache = Z
    return A, Z_cache

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    Z_cache = Z
    return A, Z_cache




##############################################################################################################################

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, Z_cache = sigmoid(Z)
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, Z_cache = relu(Z)
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


    
##############################################################################################################################

def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-8, 1 - 1e-8)
    j = -(1/m) * np.sum((Y*np.log(AL)) + ((1-Y)*np.log(1-AL)))
    j = np.squeeze(j)
    
    return j


    
##############################################################################################################################

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



##############################################################################################################################
    
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


    
##############################################################################################################################

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(params)//2
    
    for l in range(1, L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * grads["dW"+str(l)])
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * grads["db"+str(l)])

    return parameters

##############################################################################################################################

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



##############################################################################################################################

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



##############################################################################################################################

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



##############################################################################################################################


def gradient_checking(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    parameters_values, keys = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)

    
    num_parameters = len(parameters_values)
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i] += epsilon
        J_plus[i], _ = gradient_checking_forward_propagation(X, Y, vector_to_dictionary(theta_plus,parameters))

        theta_minus = np.copy(parameters_values)
        theta_minus[i] -=  epsilon
        J_minus[i], _ = gradient_checking_forward_propagation(X, Y, vector_to_dictionary(theta_minus,parameters))

        gradapprox[i] = (J_plus[i]-J_minus[i]) / (2*epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = (numerator) / (denominator)
    
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
            
    return difference


def vector_to_dictionary(theta, shape_reference):
    parameters = {}
    cursor = 0
    for l in range(1, len(shape_reference) //2 + 1):
        Wl_shape = shape_reference["W"+str(l)].shape
        bl_shape = shape_reference["b"+str(l)].shape

        size_W = Wl_shape[0]*Wl_shape[1]
        parameters["W"+str(l)] = theta[cursor: cursor+size_W].reshape(Wl_shape)
        cursor += size_W

        size_b = bl_shape[0]*bl_shape[1]
        parameters["b"+str(l)] = theta[cursor: cursor+size_b].reshape(bl_shape)
        cursor += size_b

    return parameters



def gradients_to_vector(gradients):
    
    theta = []
    L = len(gradients) // 2

    for l in range(1, L + 1):
        
        for grad in ['dW', 'db']:
            key = grad + str(l)
            vector = gradients[key].reshape(-1, 1)
            theta.append(vector)

    theta = np.concatenate(theta, axis=0)
    return theta


def dictionary_to_vector(parameters):

    keys = []
    theta = []
    
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        
        for param in ['W', 'b']:
            key = param + str(l)
            vector = parameters[key].reshape(-1, 1)
            keys += [key] * vector.shape[0]
            theta.append(vector)
    
    theta = np.concatenate(theta, axis=0)
    return theta, keys



def gradient_checking_backward_propagation(X, Y, cache):
    grads = {}
    L = len(cache)//4
    m = X.shape[1]
    AL = cache[-3]
    Y = Y.reshape(AL.shape)

    dZL = AL - Y
    dWL = (1 / m) * np.dot(dZL, cache[-7].T)
    dbL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
    dA_prev = np.dot(cache[-2].T, dZL)

    # grads["dA" + str(L-1)] = dA_prev
    # grads["dZ" + str(L)] = dZL
    grads["dW" + str(L)] = dWL
    grads["db" + str(L)] = dbL

    for l in reversed(range(L-1)):
        dA_current = dA_prev
        
        Zl = cache[((l+1)*4)-4]
        Al = cache[((l+1)*4)-3]
        Wl = cache[((l+1)*4)-2]
        bl = cache[((l+1)*4)-1]
        
        if l > 0:
            A_prev = cache[l*4-3]
        else:
            A_prev = X
        
        dZl = np.multiply(dA_current, np.int64(Al > 0))
        
        dA_prev, dWl, dbl = linear_activation_backward(dA_current, ((A_prev, Wl, bl), Zl) ,"relu")
        
        # grads["dA" + str(l)] = dA_prev
        # grads["dZ" + str(l+1)] = dZl
        grads["dW" + str(l+1)] = dWl
        grads["db" + str(l+1)] = dbl    

    return grads


def gradient_checking_forward_propagation(X, Y, parameters):
    m = X.shape[1]
    caches = []
    L = len(parameters)//2
    A = X

    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches += [cache[1], A, cache[0][1], cache[0][2]]

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches += [cache[1], AL, cache[0][1], cache[0][2]]

    cost = (1./m) * np.sum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1-AL), 1-Y))
    
    caches = tuple(caches)
    return cost, caches


def forward_propagation(X, parameters):
    m = X.shape[1]
    caches = []
    L = len(parameters)//2
    A = X

    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches += [cache[1], A, cache[0][1], cache[0][2]]

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches += [cache[1], AL, cache[0][1], cache[0][2]]
    
    caches = tuple(caches)
    return AL, caches


##############################################################################################################################


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    Y = Y.reshape(1, m)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    total_mini_batches = math.floor(m/mini_batch_size)

    for k in range(total_mini_batches):
        batch_X = shuffled_X[:, k*mini_batch_size:((k+1)*mini_batch_size)]
        batch_Y = shuffled_Y[:, k*mini_batch_size:((k+1)*mini_batch_size)]
        
        mini_batch = (batch_X, batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        j = (m % mini_batch_size)
        last_batch_X = shuffled_X[:, -j:]
        last_batch_Y = shuffled_Y[:, -j:]
        
        last_mini_batch = (last_batch_X, last_batch_Y)
        mini_batches.append(last_mini_batch)
    return mini_batches


##############################################################################################################################


def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(1, L+1):
        v["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L=len(parameters)//2

    for l in range(1, L+1):
        v["dW"+str(l)] = (beta*v["dW"+str(l)]) + ((1-beta)*grads["dW"+str(l)])
        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * v["dW"+str(l)])
        v["db"+str(l)] = (beta*v["db"+str(l)]) + ((1-beta)*grads["db"+str(l)])
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * v["db"+str(l)])

    return parameters, v


##############################################################################################################################


def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(1, L+1):
        v["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        s["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)
        s["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate= 0.01, beta1=0.9, beta2=0.999, epsilon= 1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        v["dW"+str(l)] = (beta1*v["dW"+str(l)]) + ((1-beta1)*grads["dW"+str(l)])
        v["db"+str(l)] = (beta1*v["db"+str(l)]) + ((1-beta1)*grads["db"+str(l)])

        v_corrected["dW"+str(l)] = v["dW"+str(l)] / (1-(beta1)**t)
        v_corrected["db"+str(l)] = v["db"+str(l)] / (1-(beta1)**t)
        
        s["dW"+str(l)] = (beta2*s["dW"+str(l)]) + ((1-beta2)*grads["dW"+str(l)]**2)
        s["db"+str(l)] = (beta2*s["db"+str(l)]) + ((1-beta2)*grads["db"+str(l)]**2)

        s_corrected["dW"+str(l)] = s["dW"+str(l)] / (1-(beta2)**t)
        s_corrected["db"+str(l)] = s["db"+str(l)] / (1-(beta2)**t)

        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * (v_corrected["dW"+str(l)] / (np.sqrt(s_corrected["dW"+str(l)])+epsilon)))
        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * (v_corrected["db"+str(l)] / (np.sqrt(s_corrected["db"+str(l)])+epsilon)))

    
    return parameters, v, s, v_corrected, s_corrected

##############################################################################################################################


def update_lr(learning_r, epoch, decay_rate):
    learning_rate = 1/(1+decay_rate*epoch)*learning_r
    return learning_rate

def schedule_lr_decay(learning_r, epoch, decay_rate, time_interval=1000):
    learning_rate = 1/(1+decay_rate*np.floor(epoch/time_interval))*learning_r
    return learning_rate