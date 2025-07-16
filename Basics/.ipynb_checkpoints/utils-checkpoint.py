import numpy as np

### ex1
def update_parameters_with_gd_test_case():
    np.random.seed(1)
    learning_rate = 0.01
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,2)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,2)
    db2 = np.random.randn(3,1)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return parameters, grads, learning_rate

"""
def update_parameters_with_sgd_checker(function, inputs, outputs):
    if function(inputs) == outputs:
        print("Correct")
    else:
        print("Incorrect")
"""


### ex 2
def random_mini_batches_test_case():
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    return X, Y, mini_batch_size


### ex 3
def initialize_velocity_test_case():
    np.random.seed(1)
    W1 = np.random.randn(3,2)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


### ex 4
def update_parameters_with_momentum_test_case():
    np.random.seed(1)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,2)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,2)
    db2 = np.random.randn(3,1)
   
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    v = {'dW1': np.array([[ 0.,  0.,  0.],
                          [ 0.,  0.,  0.]]), 
         'dW2': np.array([[ 0.,  0.],
                          [ 0.,  0.],
                          [ 0.,  0.]]), 
         'db1': np.array([[ 0.],
                          [ 0.]]), 
         'db2': np.array([[ 0.],
                          [ 0.],
                          [ 0.]])}
    
    return parameters, grads, v


### ex 5
def initialize_adam_test_case():
    np.random.seed(1)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,2)
    b2 = np.random.randn(3,1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


### ex 6
def update_parameters_with_adam_test_case():
    np.random.seed(1)
    v, s = ({'dW1': np.array([[ 0.,  0.,  0.], # (2, 3)
                              [ 0.,  0.,  0.]]), 
             'dW2': np.array([[ 0.,  0.],      # (3, 2)
                              [ 0.,  0.],
                              [ 0.,  0.]]), 
             'db1': np.array([[ 0.],           # (2, 1)
                              [ 0.]]), 
             'db2': np.array([[ 0.],          # (3, 1)
                              [ 0.],
                              [ 0.]])}, 
            {'dW1': np.array([[ 0.,  0.,  0.], # (2, 3)
                              [ 0.,  0.,  0.]]), 
             'dW2': np.array([[ 0.,  0.],      # (3, 2)
                              [ 0.,  0.],
                              [ 0.,  0.]]), 
             'db1': np.array([[ 0.],           # (2, 1)
                              [ 0.]]), 
             'db2': np.array([[ 0.],           # (3, 1)
                              [ 0.],
                              [ 0.]])})
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,2)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,2)
    db2 = np.random.randn(3,1)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    t = 2
    learning_rate = 0.02
    beta1 = 0.8
    beta2 = 0.888
    epsilon = 1e-2
    
    return parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon




















from numpy import array
from dlai_tools.testing_utils import single_test, multiple_test

### ex 1         
def update_parameters_with_gd_test(target):
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    learning_rate = 0.01
    
    expected_output = {'W1': np.array([[ 1.63312395, -0.61217855, -0.5339999],
                                       [-1.06196243,  0.85396039, -2.3105546]]),
                       'b1': np.array([[ 1.73978682],
                                       [-0.77021546]]),
                       'W2': np.array([[ 0.32587637, -0.24814147],
                                       [ 1.47146563, -2.05746183],
                                       [-0.32772076, -0.37713775]]),
                       'b2': np.array([[ 1.13773698],
                                       [-1.09301954],
                                       [-0.16397615]])}

    params_up = target(parameters, grads, learning_rate)

    for key in params_up.keys():
        assert type(params_up[key]) == np.ndarray, f"Wrong type for {key}. We expected np.ndarray, but got {type(params_up[key])}"
        assert params_up[key].shape == parameters[key].shape, f"Wrong shape for {key}. {params_up[key].shape} != {parameters[key].shape}"
        assert np.allclose(params_up[key], expected_output[key]), f"Wrong values for {key}. Check the formulas. Expected: \n {expected_output[key]}"
    
    print("\033[92mAll tests passed")
            
### ex 2        
def random_mini_batches_test(target):
    np.random.seed(1)
    mini_batch_size = 2
    X = np.random.randn(5, 7)
    Y = np.random.randn(1, 7) < 0.5

    expected_output = [(np.array([[ 1.74481176, -0.52817175],
                                  [-0.38405435, -0.24937038],
                                  [-1.10061918, -0.17242821],
                                  [-0.93576943,  0.50249434],
                                  [-0.67124613, -0.69166075]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[-0.61175641, -1.07296862],
                                  [ 0.3190391 ,  1.46210794],
                                  [-1.09989127, -0.87785842],
                                  [ 0.90159072,  0.90085595],
                                  [ 0.53035547, -0.39675353]]), 
                        np.array([[ True, False]])), 
                       (np.array([[ 1.62434536, -2.3015387 ],
                                  [-0.7612069 , -0.3224172 ],
                                  [ 1.13376944,  0.58281521],
                                  [ 1.14472371, -0.12289023],
                                  [-0.26788808, -0.84520564]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[ 0.86540763],
                                  [-2.06014071],
                                  [ 0.04221375],
                                  [-0.68372786],
                                  [-0.6871727 ]]), 
                        np.array([[False]]))]
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

    
### ex 3    
def initialize_velocity_test(target):
    parameters = initialize_velocity_test_case()
    
    expected_output = {'dW1': np.array([[0., 0.],
                                        [0., 0.],
                                        [0., 0.]]), 
                       'db1': np.array([[0.],
                                        [0.],
                                        [0.]]), 
                       'dW2': np.array([[0., 0., 0.],
                                        [0., 0., 0.],
                                        [0., 0., 0.]]), 
                       'db2': array([[0.],
                                     [0.],
                                     [0.]])}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

### ex 4
def update_parameters_with_momentum_test(target):
    parameters, grads, v = update_parameters_with_momentum_test_case()
    beta = 0.9
    learning_rate = 0.01
    
    expected_parameters = {'W1': np.array([[ 1.62522322, -0.61179863, -0.52875457],
                                           [-1.071868,    0.86426291, -2.30244029]]),
                           'b1': np.array([[ 1.74430927],
                                           [-0.76210776]]),
                           'W2': np.array([[ 0.31972282, -0.24924749],
                                           [ 1.46304371, -2.05987282],
                                           [-0.32294756, -0.38336269]]),
                           'b2': np.array([[ 1.1341662 ],
                                           [-1.09920409],
                                           [-0.171583  ]])}
    
    expected_v = {'dW1': np.array([[-0.08778584,  0.00422137,  0.05828152],
                                   [-0.11006192,  0.11447237,  0.09015907]]),
                  'dW2': np.array([[-0.06837279, -0.01228902],
                                   [-0.09357694, -0.02678881],
                                   [ 0.05303555, -0.06916608]]),
                  'db1': np.array([[0.05024943],
                                   [0.09008559]]),
                  'db2': np.array([[-0.03967535],
                                   [-0.06871727],
                                   [-0.08452056]])}
    
    expected_output = (expected_parameters, expected_v)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)    

    
### ex 5   
def initialize_adam_test(target):
    parameters = initialize_adam_test_case()
    
    expected_v = {'dW1': np.array([[0., 0., 0.],
                                   [0., 0., 0.]]),
                  'db1': np.array([[0.],
                                   [0.]]),
                  'dW2': np.array([[0., 0.],
                                   [0., 0.],
                                   [0., 0.]]),
                  'db2': np.array([[0.],
                                   [0.],
                                   [0.]])}
    
    expected_s = {'dW1': np.array([[0., 0., 0.],
                                   [0., 0., 0.]]),
                  'db1': np.array([[0.],
                                   [0.]]),
                  'dW2': np.array([[0., 0.],
                                   [0., 0.],
                                   [0., 0.]]),
                  'db2': np.array([[0.],
                                   [0.],
                                   [0.]])}
    
    expected_output = (expected_v, expected_s)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
### ex 6    
def update_parameters_with_adam_test(target):
    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()

    c1 = 1.0 / (1 - beta1**t)
    c2 = 1.0 / (1 - beta2**t)
    
    expected_v = {'dW1': np.array([-0.17557168,  0.00844275,  0.11656304]), 
                  'dW2': np.array([-0.13674557, -0.02457805]), 
                  'db1': np.array([0.10049887]), 
                  'db2': np.array([-0.07935071])}
    
    expected_s = {'dW1': np.array([0.08631117, 0.00019958, 0.03804344]),
                  'dW2':np.array([0.05235818, 0.00169142]),
                  'db1':np.array([0.02828006]),
                  'db2':np.array([0.0176303 ])}
    
    expected_parameters = {'W1': np.array([ 1.63937725, -0.62327448, -0.54308727]),
                           'W2':np.array([ 0.33400549, -0.23563857]),
                           'b1':np.array([ 1.72995096]),
                           'b2':np.array([ 1.14852557])}

    parameters, v, s, vc, sc  = target(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)
    
    for key in v.keys():
        
        assert type(v[key]) == np.ndarray, f"Wrong type for v['{key}']. Expected np.ndarray"
        assert v[key].shape == vi[key].shape, f"Wrong shape for  v['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(v[key][0], expected_v[key]), f"Wrong values. Check you formulas for v['{key}']"
        #print(f"v[\"{key}\"]: \n {str(v[key][0])}")

    for key in vc.keys():
        assert type(vc[key]) == np.ndarray, f"Wrong type for v_corrected['{key}']. Expected np.ndarray"
        assert vc[key].shape == vi[key].shape, f"Wrong shape for  v_corrected['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(vc[key][0], expected_v[key] * c1), f"Wrong values. Check you formulas for v_corrected['{key}']"
        #print(f"vc[\"{key}\"]: \n {str(vc[key])}")

    for key in s.keys():
        assert type(s[key]) == np.ndarray, f"Wrong type for s['{key}']. Expected np.ndarray"
        assert s[key].shape == si[key].shape, f"Wrong shape for  s['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(s[key][0], expected_s[key]), f"Wrong values. Check you formulas for s['{key}']"
        #print(f"s[\"{key}\"]: \n {str(s[key])}")

    for key in sc.keys():
        assert type(sc[key]) == np.ndarray, f"Wrong type for s_corrected['{key}']. Expected np.ndarray"
        assert sc[key].shape == si[key].shape, f"Wrong shape for  s_corrected['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(sc[key][0], expected_s[key] * c2), f"Wrong values. Check you formulas for s_corrected['{key}']"   
        # print(f"sc[\"{key}\"]: \n {str(sc[key])}")

    for key in parameters.keys():
        assert type(parameters[key]) == np.ndarray, f"Wrong type for parameters['{key}']. Expected np.ndarray"
        assert parameters[key].shape == parametersi[key].shape, f"Wrong shape for  parameters['{key}']. The update must keep the dimensions of parameters inputs"
        assert np.allclose(parameters[key][0], expected_parameters[key]), f"Wrong values. Check you formulas for parameters['{key}']"   
        #print(f"{key}: \n {str(parameters[key])}")

    print("\033[92mAll tests passed")


### ex 7    
def update_lr_test(target):
    learning_rate = 0.5
    epoch_num = 2
    decay_rate = 1
    expected_output = 0.16666666666666666
    
    output = target(learning_rate, epoch_num, decay_rate)
    
    assert np.isclose(output, expected_output), f"output: {output} expected: {expected_output}"
    print("\033[92mAll tests passed")


### ex 8    
def schedule_lr_decay_test(target):
    learning_rate = 0.5
    epoch_num_1 = 100
    epoch_num_2 = 10
    decay_rate = 1
    time_interval = 100
    expected_output_1 = 0.25
    expected_output_2 = 0.5
    
    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"
    
    learning_rate = 0.3
    epoch_num_1 = 1000
    epoch_num_2 = 100
    decay_rate = 0.25
    time_interval = 100
    expected_output_1 = 0.085714285
    expected_output_2 = 0.24

    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"

    print("\033[92mAll tests passed")





























import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def load_params_and_grads(seed=1):
    np.random.seed(seed)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2


def initialize_parameters1(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
                    
    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], 1
        
    return parameters


def compute_cost(a3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function without dividing by number of training examples
    
    Note: 
    This is used with mini-batches, 
    so we'll first accumulate costs over an entire epoch 
    and then divide by the m training examples
    """
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost_total =  np.sum(logprobs)
    
    return cost_total

def forward_propagation1(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    
    return a3, cache

def backward_propagation1(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    
    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients

def predict1(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int64)
    
    # Forward propagation
    a3, caches = forward_propagation1(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y

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
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y












