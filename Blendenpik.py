import numpy as np
import scipy as sp
import scipy.io
import random as rn
import math
import time
from scipy import random, linalg, diag, sparse
from scipy.fftpack import dct
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import lsqr
from random import uniform, randrange

m = 30000
n = 750
gamma = 1.5
p = gamma * n / m

A = sp.sparse.rand(m,n, density=.1, format="lil")
x_soln = np.random.rand(n) # the true x to solve Ax = b
b = A*x_soln

# Start Blendenpik
start_time = time.time()

# #_m (~m) is the next multiple of 1000
_m = math.ceil(m / 1000) * 1000
# M is the matrix A with ~m-m rows of 0 padded on
M = np.vstack( (A.toarray(), np.zeros((_m - m , n))) )

# Apply D to M
D = scipy.sparse.diags(np.random.choice([-1,1], _m))   
# Apply Discrete Cosine Transform (F) of D(M)
M = sp.fftpack.dct(D*M)
M[0] = M[0] / math.sqrt(2)

# randomly select rows with probability p = gamma * n /m
p = min(1, gamma*n/_m)
selectedRows = np.random.choice(_m, int(p*_m))
SM = M[selectedRows,:]

# QR factorization of S(F(D(M)))
Q, R = np.linalg.qr(SM)

# Solves the related system
y = sp.sparse.linalg.lsqr(A * np.linalg.inv(R), b)[0]
#solves for the final answer
x = sp.sparse.linalg.lsqr(R, y)[0]

end_time = time.time()

# Directly solves the system
start_time1 = time.time()
x_ = sp.sparse.linalg.lsqr(A, b)[0]
end_time1 = time.time()

n1 = np.linalg.norm(x-x_soln,ord = 2)
n2 = np.linalg.norm(x_-x_soln,ord = 2)

print("Condition number of A - ", np.linalg.cond(A.toarray()))
print("2-Norm difference - Blendenpik estimate\t", n1, " in ", end_time - start_time)
print("2-Norm difference - scipy lsqr\t\t", n2, " in ", end_time1 - start_time1)
print("Time ratio, numpy/Blendenpik:", (end_time1 - start_time1)/(end_time - start_time))

