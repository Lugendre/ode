use anyhow::Result;
use eom::{adaptor, explicit, Scheme};
use itertools::unfold;
use kepler::implicit::RKGL36;
use kepler::test::Test;
use kepler::Kepler;
use ndarray::arr1;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::prelude::{IntoDrawingArea, WHITE};
use plotters::series::LineSeries;
use plotters::style::RED;
use std::iter::from_fn;

fn main() -> Result<()> {
    // Kepler's orbit
    // t=20までの軌道を計算する
    // let e = 0.9;
    // let dt = 0.1;
    // let eom = Kepler::default();
    // // let mut teo = explicit::RK4::new(eom, dt);
    // let mut teo = RKGL36::new(eom, dt);
    // let x0 = 1.0 - e;
    // let y0 = 0.0;
    // let v0 = 0.0;
    // let u0 = f64::sqrt((1.0 + e) / (1.0 - e));
    // let ts = adaptor::time_series(arr1(&[x0, y0, v0, u0]), &mut teo);
    // let end_time = 2001;

    let eom = Test::default();
    let x0 = 0.0;
    let v0 = 1.0;

    let mut dt = 1.0;
    let dts = from_fn(move || {
        if dt < 1e-3 {
            None
        } else {
            dt *= 0.9;
            Some(dt)
        }
    });

    for dt in dts {
        let mut teo = RKGL36::new(eom, dt);
        let ts = adaptor::time_series(arr1(&[x0, v0]), &mut teo);
        for (n, xs) in ts.enumerate() {
            let t = n as f64 * dt;
            if t >= 100.0 {
                let x = t.sin();
                let e = ((xs[0] - x)/x).abs();
                println!("{} {}", dt, e);
                break;
            }
        }
    }

    // for (t, xs) in ts.take(end_time).enumerate() {
    //     println!("t = {}, x = {}, y = {}", t, xs[0], xs[1]);
    // }

    // for (t, xs) in ts.take(end_time).enumerate() {
    //     println!("{} {} {}", t, xs[0], xs[1]);
    // }

    // let root = BitMapBackend::new("kepler.png", (640, 480)).into_drawing_area();
    // root.fill(&WHITE)?;
    // let mut chart = ChartBuilder::on(&root)
    //     .caption("Kepler orbit", ("sans-serif", 30))
    //     .margin(10)
    //     .x_label_area_size(30)
    //     .y_label_area_size(30)
    //     // .build_cartesian_2d(-2.0..0.2, -0.5..0.5)?;
    //     .build_cartesian_2d(-0.5..-0.5, -1.5..1.5)?;
    //
    // chart.configure_mesh().draw()?;
    //
    // chart.draw_series(LineSeries::new(ts.take(end_time).map(|xs| {
    //     (xs[0], xs[1])
    // }), &RED))?;

    // let root = BitMapBackend::new("test.png", (640, 480)).into_drawing_area();
    // root.fill(&WHITE)?;
    // let mut chart = ChartBuilder::on(&root)
    //     .caption("test", ("sans-serif", 30))
    //     .margin(10)
    //     .x_label_area_size(30)
    //     .y_label_area_size(30)
    //     .build_cartesian_2d(-0.0..20.0, -1.2..1.2)?;
    //
    // chart.configure_mesh().draw()?;
    //
    // chart.draw_series(LineSeries::new(ts.take(end_time).enumerate().map(|(t, xs)| {
    //     (t as f64 * dt, xs[1])
    // }), &RED))?;

    Ok(())
}
