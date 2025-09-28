import numpy as np

def expr_func_test_1(x,y):
    gamma = 0.577215664901532
    return 2./np.pi * ( np.log(x/2.) + gamma ) + y*0.123456
