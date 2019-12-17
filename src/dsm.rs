
// else:
//     A, B2, C, D2 = zpk2ss(poles, zeros, -1)    # A realization of 1/H
//     # Transform the realization so that C = [1 0 0 ...]
//     C, D2 = np.real_if_close(C), np.real_if_close(D2)
//     Sinv = orth(np.hstack((np.transpose(C), np.eye(order)))) / norm(C)
//     S = inv(Sinv)
//     C = np.dot(C, Sinv)
//     if C[0, 0] < 0:
//         S = -S
//         Sinv = -Sinv
//     A = np.dot(np.dot(S, A), Sinv) 
//     B2 = np.dot(S, B2) 
//     C = np.hstack((np.ones((1, 1)), np.zeros((1, order-1)))) # C=C*Sinv; 
//     D2 = np.zeros((0,))
//     # !!!! Assume stf=1
//     B1 = -B2
//     D1 = 1
//     B = np.hstack((B1, B2))

N = u.shape[1]
v = np.empty((nq, N), dtype=np.float64)
y = np.empty((nq, N), dtype=np.float64)     # to store the quantizer input
xn = np.empty((order, N), dtype=np.float64) # to store the state information
xmax = np.abs(x0) # to keep track of the state maxima

for i in range(N):
    # y0 needs to be cast to real because ds_quantize needs real
    # inputs. If quantization were defined for complex numbers,
    # this cast could be removed
    y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:, i]))
    y[:, i] = y0
    v[:, i] = ds_quantize(y0, nlev)
    x0 = np.dot(A, x0) + np.dot(B, np.concatenate((u[:, i], v[:, i])))
    xn[:, i] = np.real_if_close(x0.T)
    xmax = np.max(np.hstack((np.abs(x0).reshape((-1, 1)), xmax.reshape((-1, 1)))),
                    axis=1, keepdims=True)

return v.squeeze(), xn.squeeze(), xmax, y.squeeze()

def ds_quantize(y, n):
    """v = ds_quantize(y,n)
    Quantize y to:
        
    * an odd integer in [-n+1, n-1], if n is even, or
    * an even integer in [-n, n], if n is odd.
    This definition gives the same step height for both mid-rise
    and mid-tread quantizers.
    """
    v = np.zeros(y.shape)
    for qi in range(n.shape[0]): 
        if n[qi] % 2 == 0: # mid-rise quantizer
            v[qi] = 2*np.floor(0.5*y[qi]) + 1
        else: # mid-tread quantizer
            v[qi] = 2*np.floor(0.5*(y[qi] + 1))
        L = n[qi] - 1
        v[qi] = np.sign(v[qi])*np.min((np.abs(v[qi]), L))
    return v