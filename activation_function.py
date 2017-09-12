import numpy as np
import matplotlib.pyplot as plt


#define activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def step_function(x):
    if x > 0.0:
        return 1
    else:
        return 0

def Relu(x):
    if x > 0.0:
        return x
    else:
        return 0


x = np.arange(-5.0, 5.0, 0.1)

y_step = x.copy()

y_sigmoid = sigmoid(x)

y_Relu = x.copy()


for i in range(0, 100):
    y_step[i] = step_function(x[i])
    y_Relu[i] = Relu(x[i])

#plot
plt.plot(x, y_step)
plt.plot(x, y_sigmoid)
plt.plot(x, y_Relu)
plt.ylim(-0.1, 2)

plt.show()