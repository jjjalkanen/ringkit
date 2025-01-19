use num::BigRational;
use num_bigint::BigInt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::Python;

mod linear_container;
mod matrix;
mod matrix_algorithms;
mod ring_element;
mod utils;

use matrix::Matrix;
use matrix_algorithms::{
    unnormalized_row_echelon_form_builder, Buildable, Nullspace, NullspaceBuilder, QRDecomposition,
    QRDecompositionBuilder,
};

#[pymodule]
fn hermite_rust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nullspace, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;

    Ok(())
}

#[pyfunction]
fn nullspace<'py>(
    py: Python<'py>,
    a_rows: &'py Bound<'py, PyAny>,
    a_cols: &'py Bound<'py, PyAny>,
    a_data: &'py Bound<'py, PyAny>,
    dtype: i32,
) -> PyResult<Py<pyo3::PyAny>> {
    let rows: usize = a_rows
        .extract()
        .map_err(|_| PyValueError::new_err("Expected usize rows"))?;
    let cols: usize = a_cols
        .extract()
        .map_err(|_| PyValueError::new_err("Expected usize cols"))?;

    let n_py: Bound<'_, PyList> = match dtype {
        0 => {
            let data: Vec<BigInt> = a_data
                .extract()
                .map_err(|_| PyValueError::new_err("Expected a list of integers"))?;
            let matrix_a = Matrix::new(rows, cols, data);
            let builder = NullspaceBuilder::new(matrix_a);
            let decomp = unnormalized_row_echelon_form_builder::<
                BigInt,
                Vec<BigInt>,
                Nullspace<BigInt, Vec<BigInt>>,
                NullspaceBuilder<BigInt, Vec<BigInt>>,
            >(builder);
            let base = Nullspace::build(decomp);

            let (rows, cols) = (base.rows, base.cols);

            PyList::new_bound(
                py,
                (0..rows).map(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    PyList::new_bound(
                        py,
                        base.data[start..end].iter().map(|elem| elem.to_object(py)),
                    )
                }),
            )
        }
        1 => {
            let data_tuple: Vec<Bound<'py, PyTuple>> = a_data
                .extract()
                .map_err(|_| PyValueError::new_err("Expected a list of integer tuples"))?;
            let data = data_tuple
                .into_iter()
                .map(|py_tuple: Bound<'py, PyTuple>| {
                    let (x, y) = py_tuple.extract::<(BigInt, BigInt)>()?;
                    Ok(BigRational::new(x, y))
                })
                .collect::<PyResult<Vec<BigRational>>>()?;
            let matrix_a = Matrix::new(rows, cols, data);
            let builder = NullspaceBuilder::new(matrix_a);
            let decomp = unnormalized_row_echelon_form_builder::<
                BigRational,
                Vec<BigRational>,
                Nullspace<BigRational, Vec<BigRational>>,
                NullspaceBuilder<BigRational, Vec<BigRational>>,
            >(builder);
            let base = Nullspace::build(decomp);

            let (rows, cols) = (base.rows, base.cols);

            // Reshape the Rust Vec into a list of lists (nested Python list)
            PyList::new_bound(
                py,
                (0..rows).map(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    PyList::new_bound(
                        py,
                        base.data[start..end]
                            .iter()
                            .map(|elem| (elem.numer(), elem.denom()).to_object(py)),
                    )
                }),
            )
        }
        _ => {
            panic!("Not implemented");
        }
    };

    Ok(n_py.to_object(py))
}

#[pyfunction]
fn qr<'py>(
    py: Python<'py>,
    a_rows: &'py Bound<'py, PyAny>,
    a_cols: &'py Bound<'py, PyAny>,
    a_data: &'py Bound<'py, PyAny>,
    dtype: i32,
) -> PyResult<(Py<pyo3::PyAny>, Py<pyo3::PyAny>)> {
    let rows: usize = a_rows
        .extract()
        .map_err(|_| PyValueError::new_err("Expected usize rows"))?;
    let cols: usize = a_cols
        .extract()
        .map_err(|_| PyValueError::new_err("Expected usize cols"))?;

    let (q_py, r_py): (Bound<'_, PyList>, Bound<'_, PyList>) = match dtype {
        0 => {
            let data: Vec<BigInt> = a_data
                .extract()
                .map_err(|_| PyValueError::new_err("Expected a list of integers"))?;
            let matrix_a = Matrix::new(rows, cols, data);
            let builder = QRDecompositionBuilder::new(matrix_a);
            let decomp = unnormalized_row_echelon_form_builder::<
                BigInt,
                Vec<BigInt>,
                QRDecomposition<BigInt, Vec<BigInt>>,
                QRDecompositionBuilder<BigInt, Vec<BigInt>>,
            >(builder);
            let (q, r) = QRDecomposition::build(decomp);

            (matrix_to_list(py, q), matrix_to_list(py, r))
        }
        1 => {
            let data_tuple: Vec<Bound<'py, PyTuple>> = a_data
                .extract()
                .map_err(|_| PyValueError::new_err("Expected a list of integer tuples"))?;
            let data = data_tuple
                .into_iter()
                .map(|py_tuple: Bound<'py, PyTuple>| {
                    let (x, y) = py_tuple.extract::<(BigInt, BigInt)>()?;
                    Ok(BigRational::new(x, y))
                })
                .collect::<PyResult<Vec<BigRational>>>()?;
            let matrix_a = Matrix::new(rows, cols, data);

            let builder = QRDecompositionBuilder::new(matrix_a);
            let decomp = unnormalized_row_echelon_form_builder::<
                BigRational,
                Vec<BigRational>,
                QRDecomposition<BigRational, Vec<BigRational>>,
                QRDecompositionBuilder<BigRational, Vec<BigRational>>,
            >(builder);
            let (q, r) = QRDecomposition::build(decomp);

            (matrix_to_list(py, q), matrix_to_list(py, r))
        }
        _ => {
            panic!("Not implemented");
        }
    };

    Ok((q_py.to_object(py), r_py.to_object(py)))
}

trait TransferDatatypeConvertible {
    fn convert(self, py: Python<'_>) -> Bound<'_, pyo3::PyAny>;
}

impl TransferDatatypeConvertible for BigInt {
    fn convert(self, py: Python<'_>) -> Bound<'_, pyo3::PyAny> {
        self.to_object(py).into_bound(py)
    }
}

impl TransferDatatypeConvertible for BigRational {
    fn convert(self, py: Python<'_>) -> Bound<'_, pyo3::PyAny> {
        self.into_raw().to_object(py).into_bound(py)
    }
}

fn matrix_to_list<T>(py: Python<'_>, mx: Matrix<T, Vec<T>>) -> pyo3::Bound<'_, PyList>
where
    T: Clone + TransferDatatypeConvertible,
{
    let (rows, cols) = (mx.rows, mx.cols);

    let mut mx_data = mx.data.into_iter().map(|elem| elem.convert(py));
    // Reshape the Rust Vec into a list of lists (nested Python list)
    PyList::new_bound(
        py,
        (0..rows).map(|i| {
            let start = i * cols;
            PyList::new_bound(py, (start..(start + cols)).map(|_| mx_data.next()))
        }),
    )
}
