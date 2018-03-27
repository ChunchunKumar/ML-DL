
import matplotlib.pyplot as plt
import numpy as np
import math
np.random.seed(1)
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant',constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    return X_pad
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + np.float(b)
    return Z
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) =A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H =math.floor((n_H_prev+2*pad-f)/stride+1)
    n_W = math.floor((n_W_prev+2*pad-f)/stride+1)
    Z =np.zeros((m,n_H,n_W,n_C))
    assert (Z.shape == (m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad =A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start =stride*h
                    vert_end =stride*h+f
                    horiz_start =stride*w
                    horiz_end =stride*w+f
                    a_slice_prev =a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i, h, w, c] =conv_single_step(a_slice_prev,W[:, :, :,c],b[0,0,0,c])
    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache