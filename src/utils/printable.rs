use num::{BigInt, BigRational};

use crate::linear_container::LinearContainer;
use crate::matrix::Matrix;

pub trait Printable {
    // Prints the matrix with aligned commas and minimal whitespace.
    #[allow(unused)]
    fn print_aligned(&self);
}

impl Printable for Matrix<BigRational, Vec<BigRational>> {
    fn print_aligned(&self) {
        let num_rows = self.rows;
        let num_cols = self.cols;

        for row in 0..num_rows {
            for col in 0..num_cols {
                let idx = self.index(row, col);
                let value = LinearContainer::get(&self.data, idx);
                print!("({:?},{:?})", value.numer(), value.denom());

                // Add comma if not the last column
                if col < num_cols - 1 {
                    print!(",");
                }
            }
            println!();
        }
    }
}

impl Printable for Matrix<i64, Vec<i64>> {
    fn print_aligned(&self) {
        let num_rows = self.rows;
        let num_cols = self.cols;
        println!("Dimensions: rows {num_rows}, cols {num_cols}");

        for row in 0..num_rows {
            for col in 0..num_cols {
                let idx = self.index(row, col);
                let value = self.data.get(idx);
                print!("{value:?}");

                if col < num_cols - 1 {
                    print!(",");
                }
            }
            println!();
        }
    }
}

impl Printable for Matrix<BigInt, Vec<BigInt>> {
    fn print_aligned(&self) {
        let num_rows = self.rows;
        let num_cols = self.cols;
        println!("Dimensions: rows {num_rows}, cols {num_cols}");

        for row in 0..num_rows {
            // let mut line = String::new();
            for col in 0..num_cols {
                let idx = self.index(row, col);
                let value = self.data.get(idx);
                print!("{value:?}");

                if col < num_cols - 1 {
                    print!(",");
                }
            }
            println!();
        }
    }
}
