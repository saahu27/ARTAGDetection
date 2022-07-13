import cv2 as cv2
import numpy as np


X_W = [0,0,0,0,7,0,7,0]
Y_W = [0,3,7,11,1,11,9,1]
Z_W = [0,0,0,0,0,7,0,7]


X_I = [757,758,758,759,1190,329,1204,340]
Y_I = [213,415,686,966,172,1041,850,159]

A = []
for i in range(8):
    row1 = np.array([X_W[i], Y_W[i], Z_W[i], 1,0, 0, 0, 0,-X_W[i]*X_I[i], -Y_W[i]*X_I[i], -X_I[i]*Z_W[i],-X_I[i]])
    A.append(row1)
    row2 = np.array([0, 0, 0, 0, X_W[i], Y_W[i], Z_W[i], 1, -X_W[i]*Y_I[i], -Y_W[i]*Y_I[i], -Y_I[i]*Z_W[i],-Y_I[i]])
    A.append(row2)


A = np.array(A)

U, E, VT = np.linalg.svd(A)
V = VT.transpose()
M_vertical = V[:, V.shape[1] - 1]
M = M_vertical.reshape([3,4])

P = M[0:3,0:3]   #Projection - to extract the x,y,z
P = P/P[2,2]
R,K = np.linalg.qr(P)
print(K)
