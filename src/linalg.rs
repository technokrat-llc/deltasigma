use ndarray::{
    Array,
    ArrayView,
    Ix1,
    Ix2,
    s,
    stack,
    Axis,
};
use ndarray_linalg::svd::SVD;
use num::Complex;

pub fn orth(A: &Array<f64, Ix2>) -> ArrayView<f64, Ix2> {
    let (u, s, vh) = A.svd(true, true).unwrap();
    let M = u.unwrap().shape()[0];
    let N = vh.unwrap().shape()[1];
    let rcond = std::f64::EPSILON * std::cmp::max(M, N) as f64;
    
    let tol = s.iter().cloned().fold(0./0., f64::max) * rcond;
    let num = s.iter().filter(|s| **s > tol).count();
    u.unwrap().slice(s![.., ..num])
}