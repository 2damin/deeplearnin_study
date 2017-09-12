import numpy as np
import activation_function

#make matrix
A = np.array([1, 2, 3, 4])

print(A)
print(np.ndim(A))
print(np.shape(A))

B = np.array([[1,2], [3,4], [5,6]])

print(B)
print(np.ndim(B))
print(np.shape(B))

C = np.array([[10,10,10], [5,5,5]])

print(np.dot(B, C),"\n")

#neural network example
W = np.array([[5, 5, 8, 10, 10], [1, 5, 3, 4, 5], [10, 8, 9, 7, 10], [1, 1, 5, 2, 10]])

h1 = np.dot(A, W)

print(h1)

Z1 = activation_function.sigmoid(h1)

print("Z1: \n",Z1)