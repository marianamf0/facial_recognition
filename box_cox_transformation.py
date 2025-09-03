import numpy as np 

def box_cox_transformation(value, lamb): 
    if lamb == 0: 
        return np.log(value) 
    
    return (np.power(value, lamb) - 1.0) / lamb