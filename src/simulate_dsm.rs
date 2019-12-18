#![allow(non_snake_case)]

use ndarray::{
    Array,
    ArrayView,
    array,
    Ix1,
    Ix2,
    IxDynImpl,
    s,
    stack,
    Axis,
    Dim,
};

use ndarray_linalg::{
    norm::Norm,
    solve::Inverse,
};

use num::Complex;

use itertools::izip;

use std::iter::FromIterator;

use pyo3::{
    types::PyTuple,
};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{IntoPyArray, PyArray2, PyArray1};

pub struct ZPK {
    pub z: Array<Complex<f64>, Ix1>,
    pub p: Array<Complex<f64>, Ix1>,
    pub k: f64,
}

pub enum ModulatorType {
    ABCD(Array<f64, Ix2>),
    NTF(ZPK),
}

pub fn tf2ss(
    num: &Array<f64, Ix1>,
    den: &Array<f64, Ix1>
) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<Complex<f64>, Ix2>, Array<f64, Ix2>) {
    // num, den = normalize(num, den)   # Strips zeros, checks arrays
    // nn = len(num.shape)
    // if nn == 1:
    //     num = asarray([num], num.dtype)
    // M = num.shape[1]
    // K = len(den)
    // if M > K:
    //     msg = "Improper transfer function. `num` is longer than `den`."
    //     raise ValueError(msg)
    // if M == 0 or K == 0:  # Null system
    //     return (array([], float), array([], float), array([], float),
    //             array([], float))

    // # pad numerator to have same number of columns has denominator
    // num = r_['-1', zeros((num.shape[0], K - M), num.dtype), num]

    // if num.shape[-1] > 0:
    //     D = atleast_2d(num[:, 0])

    // else:
    //     # We don't assign it an empty array because this system
    //     # is not 'null'. It just doesn't have a non-zero D
    //     # matrix. Thus, it should have a non-zero shape so that
    //     # it can be operated on by functions like 'ss2tf'
    //     D = array([[0]], float)

    // if K == 1:
    //     D = D.reshape(num.shape)

    //     return (zeros((1, 1)), zeros((1, D.shape[1])),
    //             zeros((D.shape[0], 1)), D)

    // frow = -array([den[1:]])
    // A = r_[frow, eye(K - 2, K - 1)]
    // B = eye(K - 1, 1)
    // C = num[:, 1:] - outer(num[:, 0], den[1:])
    // D = D.reshape((C.shape[0], B.shape[1]))

    // return A, B, C, D
    unimplemented!();
}

pub fn zpk2tf(zpk: &ZPK) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
    // z = atleast_1d(z)
    // k = atleast_1d(k)
    // if len(z.shape) > 1:
    //     temp = poly(z[0])
    //     b = zeros((z.shape[0], z.shape[1] + 1), temp.dtype.char)
    //     if len(k) == 1:
    //         k = [k[0]] * z.shape[0]
    //     for i in range(z.shape[0]):
    //         b[i] = k[i] * poly(z[i])
    // else:
    //     b = k * poly(z)
    // a = atleast_1d(poly(p))

    // # Use real output if possible.  Copied from numpy.poly, since
    // # we can't depend on a specific version of numpy.
    // if issubclass(b.dtype.type, numpy.complexfloating):
    //     # if complex roots are all complex conjugates, the roots are real.
    //     roots = numpy.asarray(z, complex)
    //     pos_roots = numpy.compress(roots.imag > 0, roots)
    //     neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
    //     if len(pos_roots) == len(neg_roots):
    //         if numpy.all(numpy.sort_complex(neg_roots) ==
    //                      numpy.sort_complex(pos_roots)):
    //             b = b.real.copy()

    // if issubclass(a.dtype.type, numpy.complexfloating):
    //     # if complex roots are all complex conjugates, the roots are real.
    //     roots = numpy.asarray(p, complex)
    //     pos_roots = numpy.compress(roots.imag > 0, roots)
    //     neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
    //     if len(pos_roots) == len(neg_roots):
    //         if numpy.all(numpy.sort_complex(neg_roots) ==
    //                      numpy.sort_complex(pos_roots)):
    //             a = a.real.copy()

    // return b, a
    
    unimplemented!();
}

pub fn zpk2ss(zpk: &ZPK) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<Complex<f64>, Ix2>, Array<f64, Ix2>) {
    let (b, a) = zpk2tf(zpk);
    tf2ss(&b, &a)
}

pub fn simulate_dsm_rs(
    u: &Array<f64, Ix2>,
    arg2: ModulatorType,
    nlev: &Array<u64, Ix1>,
    mut x0: Array<f64, Ix1>,
) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<f64, Ix1>, Array<f64, Ix2>) {
    let nu = u.shape()[0];
    let nq = nlev.shape()[0];

    let order = match &arg2 {
        // This should be ensured by the type system and only be required in rust.
        ModulatorType::ABCD(ref ABCD) => {
            // TODO:
            assert_eq!(ABCD.shape()[1], nu + ABCD.shape()[0]);
            ABCD.shape()[0] - nq
        },
        ModulatorType::NTF(ZPK { z, .. }) => z.shape()[0],
    };

    let N = u.shape()[1];
    let mut v: Array<f64, Ix2> = Array::zeros((nq, N));
    let mut y: Array<f64, Ix2> = Array::zeros((nq, N));
    let mut xn: Array<f64, Ix2> = Array::zeros((order, N));
    let mut xmax = x0.mapv(f64::abs);

    match arg2 {
        ModulatorType::ABCD(ABCD) => {
            let A = ABCD.slice(s![..order, ..order]).to_owned();
            let B = ABCD.slice(s![..order, order..order + nu + nq]).to_owned();
            let C = ABCD.slice(s![order..order + nq, ..order]).to_owned();
            let D1 = ABCD.slice(s![order..order + nq, order..order + nu]).to_owned();

            for i in 0..N {
                // y0 needs to be cast to real because ds_quantize needs real
                // inputs. If quantization were defined for complex numbers,
                // this cast could be removed
                let y0 = C.dot(&x0) + D1.dot(&u.slice(s![.., i]));
                y
                    .slice_mut(s![.., i])
                    .assign(&y0);
                v
                    .slice_mut(s![.., i])
                    .assign(&ds_quantize(&y0, &nlev));
                x0 = A.dot(&x0) + B.dot(&stack![Axis(0), u.slice(s![.., i]), v.slice(s![.., i])]);
                xn
                    .slice_mut(s![.., i])
                    .assign(&x0.t());

                let xmax_len = xmax.len();
                xmax = stack![
                    Axis(1),
                    x0.mapv(|v| v.abs()).into_shape((x0.len(), 1)).unwrap(),
                    xmax.into_shape((xmax_len, 1)).unwrap()
                ].map_axis(Axis(1), |ax| ax.iter().cloned().fold(1./0. /* inf */, f64::max));
            }


        },
        // ModulatorType::NTF(ZPK { z, p, .. }) => {
        //     // Seek a realization of -1/H
        //     let (A, B2, mut C_, D2) = zpk2ss(&ZPK { z: p, p: z, k: -1.0 });
        //     C = C_.map(|c| c.re.into());
        //     // Transform the realization so that C = [1 0 0 ...]
        //     let mut Sinv = crate::linalg::orth(&stack![Axis(1), C.t(), Array::eye(order)]).to_owned() / C.norm();
        //     let mut S = Sinv.inv().unwrap();
        //     C = C.dot(&Sinv);
        //     if C[[0, 0]] < 0.0 {
        //         S = -S;
        //         Sinv = -Sinv;
        //     }
        //     A = S.dot(&A).dot(&Sinv);
        //     B2 = S.dot(&B2);
        //     C = stack![Axis(1), array![[1.]], Array::zeros((1, order - 1))];
        //     // C=C*Sinv;
        //     // D2 = 0;
        //     // !!!! Assume stf=1
        //     B1 = -B2;
        //     D1 = array![[1.0]];
        //     B = stack![Axis(1), B1, B2];
        // }
        _ => unimplemented!(),
    }

    return (v, xn, xmax, y)
}

/// v = ds_quantize(y,n)
/// Quantize y to:
    
/// * an odd integer in [-n+1, n-1], if n is even, or
/// * an even integer in [-n, n], if n is odd.
/// This definition gives the same step height for both mid-rise
/// and mid-tread quantizers.
fn ds_quantize(y: &Array<f64, Ix1>, n: &Array<u64, Ix1>) -> Array<f64, Ix1> {
    Array::from_iter(izip!(n, y).map(|(n, y)| {
        let v = 2.0 * if n % 2 == 0 {
            (0.5 * y).floor() + 1.0
        } else {
            (0.5 * (y + 1.0)).floor()
        };

        v.signum() * v.abs().min((n - 1) as f64)
    }))
}

#[pyfunction]
/// Formats the sum of two numbers as string
fn simulate_dsm(py: Python, u: &PyArray2<f64>, arg2: &PyArray2<f64>, nlev: &PyArray1<u64>, x0: &PyArray1<f64>)
-> Py<PyTuple> {
    
    let (v, xn, xmax, y) = simulate_dsm_rs(
        &u.as_array().to_owned(),
        ModulatorType::ABCD(arg2.as_array().to_owned()),
        &nlev.as_array().to_owned(),
        x0.as_array().to_owned()
    );
    Py::new(py, PyTuple::new(py, [v.into_pyarray(py).to_owned(), xn.into_pyarray(py).to_owned(), xmax.into_pyarray(py).to_owned(), y.into_pyarray(py).to_owned()]).to_owned()).unwrap()
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn dsrs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;

    Ok(())
}