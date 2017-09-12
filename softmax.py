import numpy as np

def softmax(a):
    c = max(a)
    exp_a = np.exp(a-c)
    sum_a = np.sum(exp_a)
    y = exp_a/sum_a

    return y

a = np. array([10, 15, 250])

y = softmax(a)

print(y)