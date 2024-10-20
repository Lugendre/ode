use crate::implicit::*;
use eom::{Explicit, ModelSpec};
use ndarray::*;

#[derive(Debug, Clone, Copy)]
pub struct Test;

impl Default for Test {
    fn default() -> Self {
        Self
    }
}

impl ModelSpec for Test {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        2
    }
}

impl Explicit for Test {
    fn rhs<'a, S>(&mut self, xs: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
    {
        let x = xs[0];
        let u = xs[1];
        xs[0] = u;
        xs[1] = -x;
        xs.iter().for_each(|x| {
            if x.is_nan() || x.is_infinite() {
                panic!("Numerical Error!")
            }
        });
        xs
    }
}

impl Jacobian for Test {
    fn jacobian<'a, S, T>(
        &mut self,
        _: &ArrayBase<S, Self::Dim>,
        js: &'a mut ArrayBase<T, Ix2>,
    ) -> &'a ArrayBase<T, Ix2>
    where
        S: Data<Elem = f64>,
        T: DataMut<Elem = f64>,
    {
        js[[0, 0]] = 0.0;
        js[[0, 1]] = 1.0;
        js[[1, 0]] = -1.0;
        js[[1, 1]] = 0.0;
        js
    }
}

