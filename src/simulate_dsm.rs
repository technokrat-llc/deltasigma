// -*- coding: utf-8 -*-
// _simulateDSM_cblas.py
// Module providing a cython implementation of simulateDSM,
// based on the BLAS library.
//
// This file is part of python-deltasigma.
//
// python-deltasigma is a 1:1 Python replacement of Richard Schreier's
// MATLAB delta sigma toolbox (aka "delsigma"), upon which it is heavily based.
// The delta sigma toolbox is (c) 2009, Richard Schreier.
//
// python-deltasigma is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// LICENSE file for the licensing terms.
//
// This file is originally from `pydsm`. Little modifications have been
// performed - most prominently a bug in the ABCD matrix handling has been
// fixed. Many thanks to the original author.
//
// The original file is:
// Copyright (c) 2012, Sergio Callegari
// All rights reserved.
//
// The (future?) modifications are:
// Copyright (c) 2014, G. Venturini and the python-deltasigma contributors
//

///
/// Fast simulator for a generic delta sigma modulator using external cblas
/// =======================================================================
///

// import numpy as np
// cimport numpy as np
// import scipy as sp
// __import__('scipy.signal')
// from libc.math cimport floor, fabs

// cdef extern from "cblas.h":
//     enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
//     enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
//     void cblas_dgemv(CBLAS_ORDER order, \
//         CBLAS_TRANSPOSE TransA, int M, int N,\
//         double alpha, double *A, int lda,\
//         double *X, int incX,\
//         double beta, double *Y, int incY)
//     void cblas_dcopy(int N, double *X, int incX,\
//         double *Y, int incY)

// include '_simulateDSM_helper.pxi'

use ndarray::{
    Array,
    Ix1,
    Ix2,
};

use num::Complex;

pub struct ZKP {
    pub z: Array<Complex<u64>, Ix1>,
    pub p: Array<Complex<u64>, Ix1>,
    pub k: f64,
}

pub enum ModulatorType {
    ABCD(Array<f64, Ix2>),
    NTF(ZKP),
}

pub fn simulateDSM(
    u: &Array<usize, Ix2>,
    arg2: &ModulatorType,
    nlev: &Array<usize, Ix2>,
    x0: &Array<f64, Ix2>,
    // int store_xn=False,
    // int store_xmax=False,
    // int store_y=False
) {
    let nu = u.shape()[0];
    let nq = nlev.shape()[0];

    let order = match arg2 {
        // This should be ensured by the type system and only be required in rust.
        ModulatorType::ABCD(ABCD) => {
            // TODO:
            assert_eq!(ABCD.shape()[1], nu + ABCD.shape()[0]);
            ABCD.shape()[0] - nq
        },
        ModulatorType::NTF(ZKP { z, .. }) => z.shape()[0],
    };

    // TODO: 
    assert_eq!(x0.shape()[0], order);

    let x0_temp = Array::zeros(x0.shape());

    // cdef np.ndarray A, B1, B2, C, D1
    let mut A;
    let mut B1;
    let mut B2;
    let mut C;
    let mut D1;
    // Build ISO Model
    // note that B=hstack((B1, B2))
    match arg2 {
        ModulatorType::ABCD(ABCD) => {
            A = ABCD[0..order, 0..order];
            B1 = ABCD[0..order, order..order + nu];
            B2 = ABCD[0..order, order + nu..order + nu + nq];
            C = ABCD[order..order + nq, 0..order];
            D1 = ABCD[order..order + nq, order..order + nu];
        },
        ModulatorType::NTF(ZKP { z, .. }) => {
            // Seek a realization of -1/H
            // A, B2, C, D2 = sp.signal.zpk2ss(ntf_p, ntf_z, -1)
            // C=C.real
            // // Transform the realization so that C = [1 0 0 ...]
            // Sinv = sp.linalg.orth(np.hstack((np.transpose(C), np.eye(order))))/ \
            //     np.linalg.norm(C)
            // S = sp.linalg.inv(Sinv)
            // C = np.dot(C, Sinv)
            // if C[0, 0] < 0:
            //     S = -S
            //     Sinv = -Sinv
            // A = np.asarray(S.dot(A).dot(Sinv), dtype=np.float64, order='C')
            // B2 = np.asarray(np.dot(S, B2), dtype=np.float64, order='C')
            // C = np.asarray(np.hstack(([[1.]], np.zeros((1,order-1)))),\
            //     dtype=np.float64, order='C')
            // // C=C*Sinv;
            // // D2 = 0;
            // // !!!! Assume stf=1
            // B1 = -B2
            // D1 = np.asarray(1., dtype=np.float64)
            //B = np.hstack((B1, B2))
        }
    }
    // else:
    //     // Seek a realization of -1/H
    //     A, B2, C, D2 = sp.signal.zpk2ss(ntf_p, ntf_z, -1)
    //     C=C.real
    //     // Transform the realization so that C = [1 0 0 ...]
    //     Sinv = sp.linalg.orth(np.hstack((np.transpose(C), np.eye(order))))/ \
    //         np.linalg.norm(C)
    //     S = sp.linalg.inv(Sinv)
    //     C = np.dot(C, Sinv)
    //     if C[0, 0] < 0:
    //         S = -S
    //         Sinv = -Sinv
    //     A = np.asarray(S.dot(A).dot(Sinv), dtype=np.float64, order='C')
    //     B2 = np.asarray(np.dot(S, B2), dtype=np.float64, order='C')
    //     C = np.asarray(np.hstack(([[1.]], np.zeros((1,order-1)))),\
    //         dtype=np.float64, order='C')
    //     // C=C*Sinv;
    //     // D2 = 0;
    //     // !!!! Assume stf=1
    //     B1 = -B2
    //     D1 = np.asarray(1., dtype=np.float64)
    //     //B = np.hstack((B1, B2))

    // // N is number of input samples to deal with
    // cdef int N = c_u.shape[1]
    // // v is output vector
    // cdef np.ndarray v = np.empty((nq, N), dtype=np.float64)
    // cdef np.ndarray y
    // if store_y:
    //     // Need to store the quantizer input
    //     y = np.empty((nq, N), dtype=np.float64)
    // else:
    //     y = np.empty((0,0), dtype=np.float64)
    // cdef np.ndarray xn
    // if store_xn:
    //     // Need to store the state information
    //     xn = np.empty((order, N), dtype=np.float64)
    // cdef np.ndarray xmax
    // if store_xmax:
    //     // Need to keep track of the state maxima
    //     xmax = np.abs(c_x0)
    // else:
    //     xmax = np.empty(0, dtype=np.float64)

    // // y0 is output before the quantizer
    // cdef np.ndarray y0 = np.empty(nq, dtype=np.float64)

    // cdef int i
    // for i in xrange(N):
    //     // Compute y0 = np.dot(C, c_x0) + np.dot(D1, u[:, i])
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, nq, order,\
    //         1.0, dbldata(C), order, \
    //         dbldata(c_x0), 1, \
    //         0.0, dbldata(y0), 1)
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, nq, nu,\
    //         1.0, dbldata(D1), nu, \
    //         dbldata(c_u)+i, N, \
    //         1.0, dbldata(y0), 1)
    //     if store_y:
    //         //y[:, i] = y0[:]
    //         cblas_dcopy(nq, dbldata(y0), 1,\
    //         dbldata(y)+i, N)
    //     ds_quantize(nq, dbldata(y0), 1, \
    //         intdata(c_nlev), 1, \
    //         dbldata(v)+i, N)
    //     // Compute c_x0 = np.dot(A, c_x0) +
    //     //   np.dot(B, np.vstack((u[:, i], v[:, i])))
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, order, order,\
    //         1.0, dbldata(A), order, \
    //         dbldata(c_x0), 1,\
    //         0.0, dbldata(c_x0_temp), 1)
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, order, nu,\
    //         1.0, dbldata(B1), nu, \
    //         dbldata(c_u)+i, N, \
    //         1.0, dbldata(c_x0_temp), 1)
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, order, nq,\
    //         1.0, dbldata(B2), nq, \
    //         dbldata(v)+i, N, \
    //         1.0, dbldata(c_x0_temp), 1)
    //     // c_x0[:,1] = c_x0_temp[:,1]
    //     cblas_dcopy(order, dbldata(c_x0_temp), 1,\
    //         dbldata(c_x0), 1)
    //     if store_xn:
    //         // Save the next state
    //         //xn[:, i] = c_x0
    //         cblas_dcopy(order, dbldata(c_x0), 1,\
    //         dbldata(xn)+i, N)
    //     if store_xmax:
    //         // Keep track of the state maxima
    //         // xmax = np.max((np.abs(x0), xmax), 0)
    //         track_vabsmax(order, dbldata(xmax), 1,\
    //             dbldata(c_x0), 1)
    // if not store_xn:
    //     xn = c_x0
    // return v.squeeze(), xn.squeeze(), xmax, y.squeeze()

}