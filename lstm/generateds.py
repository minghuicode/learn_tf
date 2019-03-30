#coding:utf-8
'''
this code generateds y=sinx sequence samples
'''
import numpy as np 
seed = 2

TIME_STEPS = 10
TRAINING_EXAMPLES = 10000
TEST_EXAMPLES = 1000
SAMPLE_GAP = 0.01 #  seq: sin(x),sin(x+0.01),...

def generateds(num_start,num_end,num_samples):
    '''
    input is X: sin(0),sin(0.01),...sin(0.09)
    output is Y: sin(0.10)
    '''
    seq = np.sin(np.linspace(num_start,num_end,num_samples,dtype=np.float32))
    X = []
    Y = []
    for i in range(len(seq)-TIME_STEPS):
        X.append([seq[i:i+TIME_STEPS]])
        Y.append([seq[i+TIME_STEPS]])
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)

def generateds_train():
    test_start = (TRAINING_EXAMPLES+TIME_STEPS)*SAMPLE_GAP
    test_end = test_start+(TEST_EXAMPLES+TIME_STEPS)*SAMPLE_GAP
    return generateds(0,test_start,TRAINING_EXAMPLES+TIME_STEPS)

def generateds_test():
    test_start = (TRAINING_EXAMPLES+TIME_STEPS)*SAMPLE_GAP
    test_end = test_start+(TEST_EXAMPLES+TIME_STEPS)*SAMPLE_GAP
    return generateds(test_start,test_end,TEST_EXAMPLES+TIME_STEPS)