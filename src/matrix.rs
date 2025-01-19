use std::marker::PhantomData;
use std::ops::{Mul, Neg};

use crate::linear_container::LinearContainer;
use crate::ring_element::RingElement;

use crate::utils::verify_features::DebugIfVerify;

#[deny(dead_code)]
#[derive(Clone)]
#[cfg_attr(feature = "verify-matrices", derive(Debug, PartialEq))]
pub struct Matrix<T: Clone, C: Clone> {
    pub rows: usize,
    pub cols: usize,
    pub data: C,
    _marker: PhantomData<T>,
}

impl<T, C> Matrix<T, C>
where
    T: RingElement,
    C: LinearContainer<T>,
{
    pub fn new(rows: usize, cols: usize, data: C) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data,
            _marker: PhantomData,
        }
    }

    /// Consumes the Matrix and returns the underlying data container
    pub fn into_data(self) -> C {
        self.data
    }

    pub fn identity(size: usize) -> Self {
        let mut data = C::with_size(T::zero(), size * size);

        // Diagonal
        for i in 0..size {
            let idx = i * size + i;
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx < data.len());
            }
            *data.get_mut(idx) = T::one();
        }

        // Convert the data vector into the container type and construct the matrix
        Matrix {
            rows: size,
            cols: size,
            data,
            _marker: PhantomData,
        }
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn set(&mut self, row: usize, col: usize, value: &T) {
        let idx = self.index(row, col);
        #[cfg(feature = "verify-indexes")]
        {
            assert!(idx < self.data.len());
        }
        *self.data.get_mut(idx) = value.clone();
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        let idx = self.index(row, col);
        #[cfg(feature = "verify-indexes")]
        {
            assert!(idx < self.data.len());
        }
        self.data.get(idx)
    }

    /// Find the pivot row and value for column k
    pub fn pivot_row(&self, edge_row: usize, k: usize) -> Option<usize> {
        return match ((edge_row + 1)..self.rows)
            .filter(|&i| {
                let idx = self.index(i, k);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx < self.data.len());
                }
                !self.data.get(idx).is_zero()
            })
            .min_by_key(|&i| {
                let idx = self.index(i, k);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx < self.data.len());
                }
                self.data.get(idx).abs()
            }) {
            Some(row) => {
                let edge_value = {
                    let idx = self.index(edge_row, k);
                    #[cfg(feature = "verify-indexes")]
                    {
                        assert!(idx < self.data.len());
                    }
                    self.data.get(idx)
                };
                let row_value = {
                    let idx = self.index(row, k);
                    #[cfg(feature = "verify-indexes")]
                    {
                        assert!(idx < self.data.len());
                    }
                    self.data.get(idx)
                };
                if row_value.abs() < edge_value.abs() || edge_value.is_zero() {
                    Some(row)
                } else {
                    Some(edge_row)
                }
            }
            None => None,
        };
    }

    pub fn null_bottom_row_count(&self) -> usize {
        let mut k = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = self.index(self.rows - i - 1, j);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx < self.data.len());
                }
                if !self.data.get(idx).is_zero() {
                    return k;
                }
            }
            k += 1;
        }

        k
    }

    pub fn null_row_count(&self) -> usize {
        let mut k = 0;
        for i in 0..self.rows {
            let mut is_null = true;
            for j in 0..self.cols {
                let idx = self.index(self.rows - i - 1, j);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx < self.data.len());
                }
                if !self.data.get(idx).is_zero() {
                    is_null = false;
                    break;
                }
            }
            if is_null {
                k += 1;
            }
        }

        k
    }

    /// Transpose the matrix using block-wise cache-aware algorithm.
    pub fn transpose(&self) -> Matrix<T, C> {
        let default_block_size: usize = usize::max(256 / size_of::<T>(), 1);

        let mut transposed_data = C::with_size(T::zero(), self.rows * self.cols);

        let m = self.rows;
        let n = self.cols;

        for i in (0..m).step_by(default_block_size) {
            let max_i = usize::min(i + default_block_size, m);
            for j in (0..n).step_by(default_block_size) {
                let max_j = usize::min(j + default_block_size, n);
                for ii in i..max_i {
                    for jj in j..max_j {
                        let src_idx = self.index(ii, jj);
                        #[cfg(feature = "verify-indexes")]
                        {
                            assert!(src_idx < self.data.len());
                        }
                        let dst_idx = jj * m + ii;
                        #[cfg(feature = "verify-indexes")]
                        {
                            assert!(dst_idx < transposed_data.len());
                        }
                        *transposed_data.get_mut(dst_idx) = self.data.get(src_idx).clone();
                    }
                }
            }
        }

        Matrix::new(self.cols, self.rows, transposed_data)
    }

    pub fn extract_rightmost_columns(&self, k: usize) -> Matrix<T, C> {
        let m = self.rows;
        let n = self.cols;

        #[cfg(feature = "verify-indexes")]
        {
            assert!(k <= n);
        }

        let mut sliced_data = C::with_size(T::zero(), m * k);
        #[cfg(feature = "verify-indexes")]
        {
            assert_eq!(sliced_data.len(), m * k);
        }

        for i in 0..m {
            for j in (n - k)..n {
                let src_idx = self.index(i, j);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(src_idx < self.data.len());
                }
                let dst_idx = (i * k) + j - n + k;
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(dst_idx < sliced_data.len());
                }
                *sliced_data.get_mut(dst_idx) = self.data.get(src_idx).clone();
            }
        }

        Matrix::new(self.rows, k, sliced_data)
    }

    pub fn extract_top_rows(&self, k: usize) -> Matrix<T, C> {
        let n = self.cols;

        #[cfg(feature = "verify-indexes")]
        {
            assert!(k <= self.rows);
        }

        let mut sliced_data = C::with_size(T::zero(), k * n);

        for i in 0..k {
            for j in 0..n {
                let idx = self.index(i, j);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx < self.data.len());
                }
                *sliced_data.get_mut(idx) = self.data.get(idx).clone();
            }
        }

        Matrix::new(k, self.cols, sliced_data)
    }
}

impl<T, C> Matrix<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
{
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        let idx1_start = self.index(row1, 0);
        let idx2_start = self.index(row2, 0);
        let length = self.cols;

        self.data.swap(idx1_start, idx2_start, length);
    }

    pub fn reduce_rows_below(&mut self, row_index: usize, col_index: usize) -> Vec<T> {
        let divisor_idx = self.index(row_index, col_index);
        #[cfg(feature = "verify-indexes")]
        {
            assert!(divisor_idx < self.data.len());
        }
        let divisor = self.data.get(divisor_idx);
        #[cfg(feature = "verify-assumptions")]
        {
            assert!(divisor != &T::zero());
        }

        let quot_values: Vec<T> = (row_index + 1..self.rows)
            .map(|i| {
                let element_idx = self.index(i, col_index);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(element_idx < self.data.len());
                }
                let element = self.data.get(element_idx);
                T::quot(element, divisor)
            })
            .collect();

        let idx2_start = self.index(row_index, row_index);
        let length = self.cols - row_index;

        for (offset, quot) in quot_values.iter().enumerate() {
            let i = row_index + 1 + offset;
            let idx1_start = self.index(i, row_index);
            let negated_quot = &quot.clone().neg();

            self.data
                .add_assign(idx1_start, idx2_start, length, negated_quot);
        }

        quot_values
    }

    pub fn reduce_rows_above(&mut self, row_index: usize, col_index: usize) -> Vec<T> {
        let divisor_idx = self.index(row_index, col_index);
        #[cfg(feature = "verify-indexes")]
        {
            assert!(divisor_idx < self.data.len());
        }
        let divisor = self.data.get(divisor_idx);
        if divisor == &T::zero() {
            return vec![];
        }

        let quot_values: Vec<T> = (0..row_index)
            .map(|i| {
                let element_idx = self.index(i, col_index);
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(element_idx < self.data.len());
                }
                let element = self.data.get(element_idx);
                T::quot(element, divisor)
            })
            .collect();

        let idx2_start = self.index(row_index, row_index);
        let length = self.cols - row_index;

        for (offset, quot) in quot_values.iter().enumerate() {
            let idx1_start = self.index(offset, row_index);
            let negated_quot = &quot.clone().neg();

            self.data
                .add_assign(idx1_start, idx2_start, length, negated_quot);
        }

        quot_values
    }

    pub fn add_row_block_below_with_weights(&mut self, weights: &[T]) {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(weights.len() < self.rows);
        }

        let row_index = self.rows - weights.len() - 1;
        let dest = self.index(row_index, 0);

        for (offset, weight) in weights.iter().enumerate() {
            let i = row_index + 1 + offset;
            // Transformation has nonzero elements at all columns, not just >= row_index
            let src = self.index(i, 0);

            self.data.add_assign(dest, src, self.cols, weight);
        }
    }

    pub fn add_row_block_above_with_weights(&mut self, weights: &[T]) {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(weights.len() < self.rows);
        }

        let dest = self.index(weights.len(), 0);

        for (offset, weight) in weights.iter().enumerate() {
            let src = self.index(offset, 0);

            self.data.add_assign(dest, src, self.cols, weight);
        }
    }
}

// Implement Mul for Matrix<T, C>
impl<'a, 'b, T, C> Mul<&'a Matrix<T, C>> for &'b Matrix<T, C>
where
    T: RingElement,
    C: LinearContainer<T>,
{
    type Output = Matrix<T, C>;

    fn mul(self, other: &'a Matrix<T, C>) -> Matrix<T, C> {
        let default_block_size: usize = usize::max(256 / size_of::<T>(), 2);

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        #[cfg(feature = "verify-indexes")]
        {
            assert_eq!(k, other.rows);
        }

        // Initialize result matrix C with zeros
        let mut result = Matrix::new(m, n, C::with_size(T::zero(), m * n));

        let bm = default_block_size;
        let bn = default_block_size;
        let bk = default_block_size;

        for i0 in (0..m).step_by(bm) {
            let imax = usize::min(i0 + bm, m);
            for j0 in (0..n).step_by(bn) {
                let jmax = usize::min(j0 + bn, n);
                for k0 in (0..k).step_by(bk) {
                    let kmax = usize::min(k0 + bk, k);
                    for i in i0..imax {
                        for k_inner in k0..kmax {
                            let a_idx = self.index(i, k_inner);
                            let a_val = self.data.get(a_idx).clone();

                            for j in j0..jmax {
                                let b_idx = other.index(k_inner, j);
                                let b_val = other.data.get(b_idx).clone();

                                let c_idx = result.index(i, j);
                                let c_ref = result.data.get_mut(c_idx);
                                *c_ref = c_ref.clone() + a_val.clone() * b_val;
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_path() {
        #[rustfmt::skip]
        let original = Matrix::new(3, 3, vec![
            2i64, 3, 4,
            5, 6, 7,
            8, 9, 10,
        ]);

        let mut matrix = original.clone();

        let quotients = matrix.reduce_rows_below(0, 0);
        #[rustfmt::skip]
        let expected_matrix = Matrix::new(3, 3, vec![
            2, 3, 4,
            1, 0, -1,
            0, -3, -6,
        ]);
        assert_eq!(matrix.data, expected_matrix.data, "Reduce rows below");

        let expected_quotients = vec![2, 4];
        assert_eq!(quotients, expected_quotients, "Quotients");

        let unit3 = Matrix::<i64, Vec<i64>>::identity(3);
        let mut reference = unit3.clone();
        reference.set(1, 0, &-expected_quotients[0]);
        reference.set(2, 0, &-expected_quotients[1]);

        assert_eq!(
            matrix.data,
            (&reference * &original).data,
            "Matrix representation"
        );

        let mut antiref = unit3.clone();
        antiref.set(1, 0, &expected_quotients[0]);
        antiref.set(2, 0, &expected_quotients[1]);

        assert_eq!(
            unit3.data,
            (&antiref * &reference).data,
            "Matrix representation of inverse"
        );

        let mut antiref_transpose = unit3.clone();
        antiref_transpose.set(0, 1, &expected_quotients[0]);
        antiref_transpose.set(0, 2, &expected_quotients[1]);

        assert_eq!(
            antiref_transpose.data,
            antiref.transpose().data,
            "Transpose of inverse"
        );

        let mut base = unit3.clone();
        base.add_row_block_below_with_weights(&quotients);
        assert_eq!(
            base.data,
            antiref.transpose().data,
            "Cheap op matches with matrix op"
        );

        matrix.swap_rows(0, 1);
        base.swap_rows(0, 1);

        #[rustfmt::skip]
        let swapped_matrix = Matrix::new(3, 3, vec![
            1, 0, -1,
            2, 3, 4,
            0, -3, -6,
        ]);
        assert_eq!(matrix.data, swapped_matrix.data, "Swap rows");

        let weights = matrix.reduce_rows_below(0, 0);
        base.add_row_block_below_with_weights(&weights);

        #[rustfmt::skip]
        let first_column_done = Matrix::new(3, 3, vec![
            1, 0, -1,
            0, 3, 6,
            0, -3, -6,
        ]);
        assert_eq!(matrix.data, first_column_done.data, "Second reduction");

        assert_eq!(
            original.data,
            (&base.transpose() * &matrix).data,
            "Undo changes"
        );

        let quot_one = matrix.reduce_rows_below(1, 1);
        base.add_row_block_below_with_weights(&quot_one);

        #[rustfmt::skip]
        let everything_done = Matrix::new(3, 3, vec![
            1, 0, -1,
            0, 3, 6,
            0, 0, 0,
        ]);
        assert_eq!(matrix.data, everything_done.data, "Final reduction");
        assert_eq!(
            original.data,
            (&base.transpose() * &matrix).data,
            "Undo everything"
        );
    }

    #[test]
    fn test_reduction_upward() {
        #[rustfmt::skip]
        let original = Matrix::new(3, 3, vec![
         1i64, 5, 4,
            0, 2, 9,
            0, 0, 1,
        ]);

        let mut matrix = original.clone();

        let first_quotients = matrix.reduce_rows_above(1, 1);
        #[rustfmt::skip]
        let once_reduced_matrix = Matrix::new(3, 3, vec![
         1i64, 1,-14,
            0, 2,  9,
            0, 0,  1,
        ]);
        assert_eq!(matrix.data, once_reduced_matrix.data);

        let unit3 = Matrix::<i64, Vec<i64>>::identity(3);

        let mut transform = unit3.clone();
        transform.add_row_block_above_with_weights(&first_quotients);

        {
            let prod = &transform.transpose() * &matrix;
            assert_eq!(original.data, prod.data);
        }

        let second_quotients = matrix.reduce_rows_above(2, 2);
        #[rustfmt::skip]
        let twice_reduced_matrix = Matrix::new(3, 3, vec![
         1i64, 1,  0,
            0, 2,  0,
            0, 0,  1,
        ]);
        assert_eq!(matrix.data, twice_reduced_matrix.data);
        transform.add_row_block_above_with_weights(&second_quotients);

        {
            let prod = &transform.transpose() * &matrix;
            assert_eq!(original.data, prod.data);
        }
    }

    #[test]
    fn test_rank_deficient_case() {
        #[rustfmt::skip]
        let mut matrix_q = Matrix::new(11, 11, vec![
         0i64,0,0,0,0,0,1,0,0,0,0,
            0,0,0,0,0,1,0,0,0,0,0,
            0,0,0,0,1,0,0,0,0,0,1,
            0,0,0,1,0,0,0,0,0,1,0,
            0,0,1,0,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,0,0,1,0,0,
            0,0,0,0,0,0,0,0,0,1,0,
            0,0,0,0,0,0,0,0,0,0,1
        ]);

        #[rustfmt::skip]
        let mut matrix_r = Matrix::new(11, 5, vec![
      1296i64,   0,-1176,    0, -359,
            0,1296,    0,-1176,    0,
            0,   0, 1296,    0,-1176,
            0,   0,    0, 1296,    0,
            0,   0,    0,    0, 1296,
            0,   0,    0,    0,    0,
            0,   0,    0,    0,    0,
            0,   0,    0,    0,    0,
            0,   0,    0,    0, 1296,
            0,   0,    0,    0,    0,
            0,   0,    0,    0, 1536
        ]);

        let matrix_a = &(matrix_q.transpose()) * &matrix_r;

        let edge_row = 4;

        {
            let quotients = matrix_r.reduce_rows_below(edge_row, 4);
            assert_eq!(quotients, vec![0, 0, 0, 1, 0, 1]);
            matrix_q.add_row_block_below_with_weights(&quotients);
        }

        let pivot_info = matrix_r.pivot_row(edge_row, 4);
        let pivot_row = match pivot_info {
            Some(row) => {
                let value = matrix_r.get(row, 4);
                assert_ne!(value, &0);
                row
            }
            None => {
                assert!(false, "In this case there is a pivot");
                0
            }
        };
        assert_eq!(pivot_row, 10);
        matrix_r.swap_rows(pivot_row, edge_row);
        matrix_q.swap_rows(pivot_row, edge_row);
        {
            let prod = &(matrix_q.transpose()) * &matrix_r;
            assert_eq!(matrix_a.data, prod.data);
        }

        let unit11 = Matrix::<i64, Vec<i64>>::identity(11);
        {
            let matrix_r_before = matrix_r.clone();
            let quotients = matrix_r.reduce_rows_below(edge_row, 4);
            assert_eq!(quotients, vec![0, 0, 0, 0, 0, 5]);

            let mut reduction_op = unit11.clone();
            reduction_op.set(10, edge_row, &-5i64);

            {
                let prod = &reduction_op * &matrix_r_before;
                assert_eq!(matrix_r.data, prod.data);
            }

            let mut reduction_inv = unit11.clone();
            reduction_inv.set(10, edge_row, &5i64);

            {
                let prod = &reduction_inv * &reduction_op;
                assert_eq!(unit11.data, prod.data);
            }

            let mut reduction_inv_trans = unit11.clone();
            reduction_inv_trans.set(edge_row, 10, &5i64);

            assert_eq!(reduction_inv.transpose().data, reduction_inv_trans.data);

            let matrix_q_before = matrix_q.clone();
            matrix_q.add_row_block_below_with_weights(&quotients);

            {
                let prod = &reduction_inv_trans * &matrix_q_before;
                assert_eq!(matrix_q.data, prod.data);
            }
        }

        {
            let prod = &(matrix_q.transpose()) * &matrix_r;
            assert_eq!(matrix_a.data, prod.data);
        }
    }
}
