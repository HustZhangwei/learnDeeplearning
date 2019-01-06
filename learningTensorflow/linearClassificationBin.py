import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab

def generate(sample_size,mean,cov,diff,regression):
    num_class = 2
    samples_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean,cov,samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci,d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d,cov,samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    if regression == False:
        class_ind = [Y == class_number for class_number in range(num_class)]
        Y = np.asarray(np.hstack(class_ind),dtype=np.float32)
    X,Y = shuffle(X0,Y0)

    return X,Y
