import numpy as np
import scipy as sp
import random as rn
import math
import time
from scipy import random, linalg, diag, sparse
from scipy.fftpack import dct
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import lsqr
from random import uniform, randrange

m = 1500
n = 20
gamma = 4
p = gamma * n / m

A = random.randn(m, n)
b = random.randn(m)


# Change condition number
U, S, V = np.linalg.svd(A)
condition = 50000000000
jump = condition/(np.size(S) - 2)
for i in range(0, 19) :
    S[i] = jump * i + 1
S1 = np.zeros((m, n))
np.fill_diagonal(S1, S)

A = np.matmul(np.matmul(U, S1), V.T)



# Start Blendenpik
start_time = time.time()


#_m (~m) is the next multiple of 1000
_m = math.ceil(m / 1000) * 1000
# M is the matrix A with ~m-m rows of 0 padded on
M = np.vstack( (A, np.zeros((_m - m , n))) )

# Apply D to M
for i in range(0, _m) :
    M[i] = M[i] * randrange(-1, 2, 2)

# Discrete Cosine Transform of M
M = sp.fftpack.dct(M)

# Apply S to M
for i in range(0, _m) :
    if (uniform(0, 1) > p) :
        M[i] = 0

# Q R factorization of S applied to M
R = linalg.qr(M, mode='economic')[1]

# Solves the related system
y = sp.sparse.linalg.lsqr( np.matmul(A, np.linalg.inv(R)), b)[0] # need to solve "(A * R^-1) * y = b" then solve "R * x = y" to get x
# solves for the final answer
x = sp.sparse.linalg.lsqr(R, y)[0]
end_time = time.time()




# Directly solves the system
start_time1 = time.time()
x_ = sp.sparse.linalg.lsqr(A, b)[0]
end_time1 = time.time()

n1 = np.linalg.norm((np.matmul(A, x) - b))
n2 = np.linalg.norm((np.matmul(A, x_) - b))

print("Norm of Blendenpik estimate", n1, " in ", end_time - start_time)
print("Norm from numpy lsqr", n2, " in ", end_time1 - start_time1)
print("Ratio:", n1/n2)