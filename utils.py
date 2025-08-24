import numpy as np

def function_perceptron_logistic(value): 
    return np.tanh(value / 2.0)

def function_derivate_perceptron_logistic(value): 
    return 0.5 * (1.0 - value**2)

def func_sigmoid(value): 
    return 1.0 / (1.0 + np.exp(-value))

def sigmoid(value): 
    return 1.0 / (1.0 + np.exp(-value))

def derivate_sigmoid_from_output(value): 
    return value * (1.0 - value)  

def tanh(value): 
    return (1.0 - np.exp(-value)) / (1.0 + np.exp(-value))

def derivate_tanh_from_output(value): 
    return 1.0 - value**2

