import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1,1,100)
#y=2x+随机噪声
train_y = 2*train_x+np.random.randn(*train_x.shape)*0.3

plt.plot(train_x,train_y,'ro',label='Original data')
plt.legend()
plt.show()