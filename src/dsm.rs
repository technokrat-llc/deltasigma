nlev = carray(nlev)
    u = np.array(u) if not hasattr(u, 'ndim') else u
    if not max(u.shape) == np.prod(u.shape):
        warn("Multiple input delta sigma structures have had little testing.")
    if u.ndim == 1:
        u = u.reshape((1, -1))
    nu = u.shape[0]
    nq = 1 if np.isscalar(nlev) else nlev.shape[0]
    # extract poles and zeros
    if (hasattr(arg2, 'inputs') and not arg2.inputs == 1) or \
       (hasattr(arg2, 'outputs') and not arg2.outputs == 1):
            raise TypeError("The supplied TF isn't a SISO transfer function.")
    if isinstance(arg2, np.ndarray):
        ABCD = np.asarray(arg2, dtype=np.float64)
        if ABCD.shape[1] != ABCD.shape[0] + nu:
            raise ValueError('The ABCD argument does not have proper dimensions.')
        form = 1
    else:
        zeros, poles, k = _get_zpk(arg2)
        form = 2
    #raise TypeError('%s: Unknown transfer function %s' % (__name__, str(arg2)))
        
    # need to set order and form now.
    order = carray(zeros).shape[0] if form == 2 else ABCD.shape[0] - nq
    
    if not isinstance(x0, collections.Iterable):
        x0 = x0*np.ones((order,), dtype=np.float64)
    else:
        x0 = np.array(x0).reshape((-1,))
    
    if form == 1:
        A = ABCD[:order, :order]
        B = ABCD[:order, order:order+nu+nq]
        C = ABCD[order:order+nq, :order]
        D1 = ABCD[order:order+nq, order:order+nu]
    else:
        A, B2, C, D2 = zpk2ss(poles, zeros, -1)    # A realization of 1/H
        # Transform the realization so that C = [1 0 0 ...]
        C, D2 = np.real_if_close(C), np.real_if_close(D2)
        Sinv = orth(np.hstack((np.transpose(C), np.eye(order)))) / norm(C)
        S = inv(Sinv)
        C = np.dot(C, Sinv)
        if C[0, 0] < 0:
            S = -S
            Sinv = -Sinv
        A = np.dot(np.dot(S, A), Sinv) 
        B2 = np.dot(S, B2) 
        C = np.hstack((np.ones((1, 1)), np.zeros((1, order-1)))) # C=C*Sinv; 
        D2 = np.zeros((0,))
        # !!!! Assume stf=1
        B1 = -B2
        D1 = 1
        B = np.hstack((B1, B2))

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