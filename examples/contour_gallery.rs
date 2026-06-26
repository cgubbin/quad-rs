use num_complex::Complex;
use quad_rs::{Contour, ContourSegment, IndentSide};

fn describe(name: &str, contour: &Contour<f64>) {
    println!("{name}");
    println!("pieces: {}", contour.pieces().len());

    for (idx, piece) in contour.pieces().iter().enumerate() {
        match piece {
            ContourSegment::Line(line) => {
                println!("  {idx}: line       {} -> {}", line.start(), line.end());
            }
            ContourSegment::CircularArc(arc) => {
                println!(
                    "  {idx}: arc        center={}, radius={}",
                    arc.center(),
                    arc.radius()
                );
            }
        }
    }

    println!();
}

fn main() {
    let upper = Contour::upper_half_disk(2.0);

    let shifted = Contour::upper_half_disk_offset(2.0, 0.1);

    let indented = Contour::piecewise_linear(vec![Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)])
        .indent(Complex::new(0.0, 0.0), 0.2, IndentSide::Left, 1e-10);

    let multi_indent =
        Contour::piecewise_linear(vec![Complex::new(-3.0, 0.0), Complex::new(3.0, 0.0)])
            .indent(Complex::new(-1.0, 0.0), 0.2, IndentSide::Left, 1e-10)
            .indent(Complex::new(1.0, 0.0), 0.2, IndentSide::Right, 1e-10);

    describe("upper half disk", &upper);
    describe("shifted upper half disk", &shifted);
    describe("single indentation", &indented);
    describe("multiple indentations", &multi_indent);
}
