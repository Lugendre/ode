use std::ops::AddAssign;
use eom::*;
use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::*;

pub trait Jacobian: ModelSpec {
    fn jacobian<'a, S, T>(
        &mut self,
        xs: &ArrayBase<S, Self::Dim>,
        js: &'a mut ArrayBase<T, Ix2>,
    ) -> &'a ArrayBase<T, Ix2>
    where
        S: Data<Elem = Self::Scalar>,
        T: DataMut<Elem = Self::Scalar>;
}

#[derive(Debug, Clone)]
pub struct RKGL36<F: Explicit> {
    f: F,
    dt: <F::Scalar as Scalar>::Real,
    epsilon: <F::Scalar as Scalar>::Real,
    a: Array2<<F::Scalar as Scalar>::Real>,
    b: Array1<<F::Scalar as Scalar>::Real>,
    c: Array1<<F::Scalar as Scalar>::Real>,
    x_l: Array2<F::Scalar>,
    k: Array2<F::Scalar>,
    g: Array2<F::Scalar>,
    j_local: Array3<F::Scalar>,
    j: Array2<F::Scalar>,
    delta: Array<F::Scalar, F::Dim>,
}

impl<A: Scalar, F: Explicit<Scalar = A>> TimeStep for RKGL36<F> {
    type Time = A::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

impl<F: Explicit<Scalar: Lapack, Dim = Ix1> + Jacobian> Scheme for RKGL36<F> {
    type Core = F;

    fn new(f: Self::Core, dt: Self::Time) -> Self {
        let epsilon = F::Scalar::real(1e-12); // 許容誤差

        // Butcher Tableau (Gauss-Legendre 3/6)
        let a = arr2(&[
            [
                F::Scalar::real(5.0 / 36.0),
                F::Scalar::real(2.0 / 9.0 - 15.0.sqrt() / 15.0),
                F::Scalar::real(5.0 / 36.0 - 15.0.sqrt() / 30.0),
            ],
            [
                F::Scalar::real(5.0 / 36.0 + 15.0.sqrt() / 24.0),
                F::Scalar::real(2.0 / 9.0),
                F::Scalar::real(5.0 / 36.0 - 15.0.sqrt() / 24.0),
            ],
            [
                F::Scalar::real(5.0 / 36.0 + 15.0.sqrt() / 30.0),
                F::Scalar::real(2.0 / 9.0 + 15.0.sqrt() / 15.0),
                F::Scalar::real(5.0 / 36.0),
            ],
        ]);
        let b = arr1(&[
            F::Scalar::real(5.0 / 18.0),
            F::Scalar::real(4.0 / 9.0),
            F::Scalar::real(5.0 / 18.0),
        ]);
        let c = arr1(&[
            F::Scalar::real(1.0 / 2.0 - 15.0.sqrt() / 10.0),
            F::Scalar::real(1.0 / 2.0),
            F::Scalar::real(1.0 / 2.0 + 15.0.sqrt() / 10.0),
        ]);

        let n = 3 * f.model_size();
        let x_l = Array2::zeros((3, f.model_size()));
        let k = Array2::zeros((3, f.model_size()));
        let g = Array2::zeros((3, f.model_size()));
        let j_local = Array3::zeros((3, f.model_size(), f.model_size()));
        let j = Array2::zeros((n, n));
        let delta = Array::zeros(n);
        Self {
            f,
            dt,
            epsilon,
            a,
            b,
            c,
            x_l,
            k,
            g,
            j_local,
            j,
            delta,
        }
    }

    fn core(&self) -> &Self::Core {
        &self.f
    }

    fn core_mut(&mut self) -> &mut Self::Core {
        &mut self.f
    }
}

impl<F: Explicit> ModelSpec for RKGL36<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.f.model_size()
    }
}

impl<F: Explicit<Scalar: Lapack, Dim = Ix1> + Jacobian<Scalar: Lapack, Dim = Ix1>> TimeEvolution
for RKGL36<F>
{
    fn iterate<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        // xのサイズ
        let dim = self.f.model_size();

        // k_lをf(x_0)で初期化
        Zip::from(self.k.rows_mut()).for_each(|mut k| {
            k.assign(&x);
            self.f.rhs(&mut k);
        });

        // Newton-Raphson法
        loop {
            // k_lからx_lを計算
            Zip::from(self.x_l.rows_mut())
                .and(self.a.rows())
                .for_each(|mut x_l, a| {
                    x_l.assign(&x);
                    Zip::from(a).and(self.k.rows()).for_each(|&a, k| {
                        x_l.add_assign(&k.map(|&k| k.mul_real(a * self.dt)));
                    });
                });

            // g_lを計算
            Zip::from(self.g.rows_mut())
                .and(self.x_l.rows())
                .and(self.k.rows())
                .for_each(|mut g, x_l, k| {
                    g.assign(&x_l);
                    self.f.rhs(&mut g);
                    g.zip_mut_with(&k, |g, k| *g = *k - *g);
                });

            // local jacobianを計算
            Zip::from(self.j_local.outer_iter_mut())
                .and(self.x_l.rows())
                .for_each(|mut j_local, x_l| {
                    self.f.jacobian(&x_l, &mut j_local);
                });

            // jacobianを計算
            for u in (0..3 * dim) {
                for v in (0..3 * dim) {
                    let i = u / dim;
                    let j = v / dim;
                    let m = v % dim;
                    let n = u % dim;
                    self.j[[u, v]] = f_delta::<F::Scalar>(i, j) * f_delta(n, m)
                        - self.j_local[[i, n, m]].mul_real(self.a[[i, j]] * self.dt);
                }
            }

            // delta_lを計算
            self.delta = self.j.solve(&(-self.g.flatten())).expect("solve failed");
            // kをk_l+1に更新
            self.k = &self.k + &self.delta.to_shape((3, dim)).expect("shape mismatch");

            // 収束判定。一旦絶対誤差で判定している。
            if self.delta.iter().all(|delta| delta.abs() < self.epsilon) {
                break;
            }
        }

        // 確定したkを使ってxを更新
        Zip::from(self.k.rows_mut())
            .and(&self.b)
            .for_each(|mut k, &b| {
                k.map_inplace(|k| *k = k.mul_real(b * self.dt));
                x.add_assign(&k);
            });

        x
    }
}

fn f_delta<T: Scalar>(i: usize, j: usize) -> T {
    if i == j {
        num::one()
    } else {
        num::zero()
    }
}
