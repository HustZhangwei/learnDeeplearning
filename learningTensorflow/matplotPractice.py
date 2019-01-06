import numpy as np
import matplotlib.pyplot as plt
import pylab


X = np.linspace(-np.pi,np.pi,256,endpoint = True)
C,S = np.cos(X),np.sin(X)
'''
#基本绘图
plt.plot(X,C)
plt.plot(X,S)

plt.show()
'''

plt.figure(figsize=(8,6),dpi = 80)
plt.subplot(1,1,1)

plt.plot(X,C,color = "blue",linewidth = 1.0,linestyle = "-",label = "COS")
plt.plot(X,S,color = "green",linewidth = "1.0",linestyle = "-",label = "SIN")

plt.xlim(-4.0,4.0)
plt.xticks(np.linspace(-4,4,9,endpoint = True))

plt.ylim(-1.0,1.0)
plt.yticks(np.linspace(-1,1,5,endpoint = True))
plt.legend()
plt.savefig("exercise.png",dpi = 72)

plt.show()





