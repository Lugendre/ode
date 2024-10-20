use crate::implicit::*;
use eom::{Explicit, ModelSpec};
use ndarray::*;
pub mod implicit;
pub mod test;

#[derive(Debug, Clone, Copy)]
pub struct Kepler {
    pub e: f64,
}

impl Default for Kepler {
    fn default() -> Self {
        Self { e: 0.9 }
    }
}

impl Kepler {
    pub fn new(e: f64) -> Self {
        Self { e }
    }
}

impl ModelSpec for Kepler {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        4
    }
}

impl Explicit for Kepler {
    fn rhs<'a, S>(&mut self, xs: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
    {
        let x = xs[0];
        let y = xs[1];
        let v = xs[2];
        let u = xs[3];
        let r = (x * x + y * y).sqrt();
        xs[0] = v;
        xs[1] = u;
        xs[2] = -x / r.powi(3);
        xs[3] = -y / r.powi(3);
        xs.iter().for_each(|x| {
            if x.is_nan() || x.is_infinite() {
                panic!("Numerical Error!")
            }
        });
        xs
    }
}

impl Jacobian for Kepler {
    fn jacobian<'a, S, T>(
        &mut self,
        xs: &ArrayBase<S, Self::Dim>,
        js: &'a mut ArrayBase<T, Ix2>,
    ) -> &'a ArrayBase<T, Ix2>
    where
        S: Data<Elem = Self::Scalar>,
        T: DataMut<Elem = Self::Scalar>,
    {
        let x = xs[0];
        let y = xs[1];
        let r = (x * x + y * y).sqrt();
        js[[0, 0]] = 0.0;
        js[[0, 1]] = 0.0;
        js[[0, 2]] = 1.0;
        js[[0, 3]] = 0.0;
        js[[1, 0]] = 0.0;
        js[[1, 1]] = 0.0;
        js[[1, 2]] = 0.0;
        js[[1, 3]] = 1.0;
        js[[2, 0]] = (3.0 * x.powi(2) - r.powi(2)) / r.powi(5);
        js[[2, 1]] = 3.0 * x * y / r.powi(5);
        js[[2, 2]] = 0.0;
        js[[2, 3]] = 0.0;
        js[[3, 0]] = 3.0 * x * y / r.powi(5);
        js[[3, 1]] = (3.0 * y.powi(2) - r.powi(2)) / r.powi(5);
        js[[3, 2]] = 0.0;
        js[[3, 3]] = 0.0;
        js
    }
}
