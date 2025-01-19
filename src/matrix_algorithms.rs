// src/matrix_algorithms.rs

use std::ops::AddAssign;
use std::ops::{Mul, Neg};

use num::BigRational;
use num_bigint::BigInt;

use crate::linear_container::LinearContainer;
use crate::matrix::Matrix;
use crate::ring_element::RingElement;

use crate::utils::verify_features::DebugIfVerify;

#[allow(unused_imports)]
use std::marker::PhantomData;

pub trait PivotSolvable<T> {
    // Solve for the pivot variable from
    // pivot_val * v[pivot_col] + sum_others = 0.
    fn solve_pivot(&mut self, result_row: usize, pivot_col: usize, pivot_val: &T, sum_others: &T);
}

impl<C> PivotSolvable<BigRational> for Matrix<BigRational, C>
where
    C: LinearContainer<BigRational>,
{
    fn solve_pivot(
        &mut self,
        current_row: usize,
        pivot_col: usize,
        pivot_val: &BigRational,
        sum_others: &BigRational,
    ) {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(current_row * self.cols + pivot_col < self.data.len());
        }
        self.set(current_row, pivot_col, &(&sum_others.neg() / pivot_val));
    }
}

impl<C> PivotSolvable<BigInt> for Matrix<BigInt, C>
where
    C: LinearContainer<BigInt>,
{
    fn solve_pivot(
        &mut self,
        current_row: usize,
        pivot_col: usize,
        pivot_val: &BigInt,
        sum_others: &BigInt,
    ) {
        let cols = self.cols;
        #[cfg(feature = "verify-indexes")]
        {
            assert!(current_row * cols + pivot_col < self.data.len());
        }

        for col in (pivot_col + 1)..cols {
            let elem = self.get(current_row, col);
            self.set(current_row, col, &(elem * pivot_val));
        }

        self.set(current_row, pivot_col, &sum_others.neg());
    }
}

impl<C> PivotSolvable<i64> for Matrix<i64, C>
where
    C: LinearContainer<i64>,
{
    fn solve_pivot(
        &mut self,
        current_row: usize,
        pivot_col: usize,
        pivot_val: &i64,
        sum_others: &i64,
    ) {
        let cols = self.cols;
        #[cfg(feature = "verify-indexes")]
        {
            assert!(current_row * cols + pivot_col < self.data.len());
        }

        for col in (pivot_col + 1)..cols {
            let elem = self.get(current_row, col);
            self.set(current_row, col, &(elem * pivot_val));
        }

        self.set(current_row, pivot_col, &sum_others.neg());
    }
}

fn solve_nullspace_basis<T, C>(echelon_form: &Matrix<T, C>, free_vars: &[usize]) -> Matrix<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    Matrix<T, C>: PivotSolvable<T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
    T: AddAssign<T>,
{
    let n_cols = echelon_form.cols;
    let n_free_vars = free_vars.len();

    #[cfg(feature = "verify-indexes")]
    {
        assert!(n_free_vars <= n_cols);
    }

    let data = C::with_size(T::zero(), n_free_vars * n_cols);
    let mut basis_vectors = Matrix::new(n_free_vars, n_cols, data);

    if n_free_vars == 0 {
        return basis_vectors;
    }

    let n_dependent_vars = n_cols - n_free_vars;

    let pivot_vars: Vec<usize> = (0..n_cols)
        .rev()
        .filter(|var| !free_vars.contains(var))
        .collect();

    for (current_output, &free_var_idx) in free_vars.iter().enumerate() {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(current_output + free_var_idx < basis_vectors.data.len());
        }
        basis_vectors.set(current_output, free_var_idx, &T::one());
        let mut row = n_dependent_vars;
        for pivot_col in &pivot_vars {
            row -= 1;
            #[cfg(feature = "verify-indexes")]
            {
                assert!(n_cols * row + pivot_col < echelon_form.data.len());
            }
            let pivot_val = echelon_form.get(row, *pivot_col);
            #[cfg(feature = "verify-assumptions")]
            {
                assert_ne!(*pivot_val, T::zero());
            }

            // sum up the contributions of other columns in that row
            let mut sum_others = T::zero();
            for col in (pivot_col + 1)..n_cols {
                sum_others += echelon_form.get(row, col) * basis_vectors.get(current_output, col);
            }

            basis_vectors.solve_pivot(current_output, *pivot_col, pivot_val, &sum_others);
        }
        #[cfg(feature = "verify-assumptions")]
        {
            assert_eq!(row, 0);
        }
    }
    #[cfg(feature = "verify-assumptions")]
    {
        let n_rows = echelon_form.rows;
        for i in 0..n_free_vars {
            let start = i * n_cols;
            for row in 0..n_rows {
                let mut accum = T::zero();
                for j in 0..n_cols {
                    assert!(start + j < basis_vectors.data.len());
                    accum += basis_vectors.get(i, j) * echelon_form.get(row, j);
                }
                assert_eq!(T::zero(), accum);
            }
        }
    }

    basis_vectors
}

#[cfg(feature = "verify-matrices")]
struct Reference<T> {
    _data: T,
}

#[cfg(feature = "verify-matrices")]
impl<T: Clone> Reference<T> {
    fn new(src: &T) -> Self {
        Self { _data: src.clone() }
    }
}

#[cfg(feature = "verify-matrices")]
impl<T> PartialEq<&T> for Reference<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &&T) -> bool {
        self._data.eq(*other)
    }
}

#[cfg(feature = "verify-matrices")]
impl<T> std::fmt::Debug for Reference<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self._data.fmt(f)
    }
}

#[cfg(not(feature = "verify-matrices"))]
struct Reference<T> {
    _data: PhantomData<T>,
}

#[cfg(not(feature = "verify-matrices"))]
impl<T> Reference<T> {
    fn new(_src: &T) -> Self {
        Self { _data: PhantomData }
    }
}

pub trait Echelonizable {
    fn has_element_at(&self, row: usize, k: usize) -> bool;

    fn pivot_row(&self, edge_row: usize, k: usize) -> Option<usize>;

    fn reduce_down(&mut self, edge_row: usize, k: usize);

    fn advance(&mut self, edge_row: usize, k: usize) -> usize;

    fn swap(&mut self, edge_row: usize, pivot_row: usize);

    fn finalize(&mut self);
}

pub struct QRDecomposition<T, C>
where
    T: Clone,
    C: Clone,
{
    #[allow(unused)]
    invariant: Reference<Matrix<T, C>>,
    q: Matrix<T, C>,
    r: Matrix<T, C>,
}

pub struct Nullspace<T, C>
where
    T: Clone,
    C: Clone,
{
    r: Matrix<T, C>,
    free_vars: Vec<usize>,
}

impl<T, C> QRDecomposition<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
{
    fn new(src: Matrix<T, C>) -> Self {
        let invariant = Reference::<Matrix<T, C>>::new(&src);
        let q = Matrix::<T, C>::identity(src.rows);
        #[cfg(feature = "verify-matrices")]
        {
            assert_eq!(invariant, &(&q.transpose() * &src));
        }

        Self {
            invariant,
            q,
            r: src,
        }
    }
}

impl<T, C> Nullspace<T, C>
where
    T: RingElement,
    C: LinearContainer<T>,
{
    fn new(src: Matrix<T, C>) -> Self {
        Self {
            r: src,
            free_vars: Vec::<usize>::new(),
        }
    }
}

pub struct QRDecompositionBuilder<T, C>
where
    T: Clone,
    C: Clone,
{
    src: Matrix<T, C>,
}

pub struct NullspaceBuilder<T, C>
where
    T: Clone,
    C: Clone,
{
    src: Matrix<T, C>,
}

pub trait Buildable<B> {
    fn build(builder: Self) -> B;
}

impl<T, C> QRDecompositionBuilder<T, C>
where
    T: RingElement,
    C: LinearContainer<T>,
{
    pub fn new(input: Matrix<T, C>) -> Self {
        Self { src: input }
    }
}

impl<T, C> NullspaceBuilder<T, C>
where
    T: RingElement,
    C: LinearContainer<T>,
{
    pub fn new(input: Matrix<T, C>) -> Self {
        Self { src: input }
    }
}

impl<T, C> Buildable<QRDecomposition<T, C>> for QRDecompositionBuilder<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
{
    fn build(builder: Self) -> QRDecomposition<T, C> {
        QRDecomposition::new(builder.src)
    }
}

impl<T, C> Buildable<Nullspace<T, C>> for NullspaceBuilder<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
{
    fn build(builder: Self) -> Nullspace<T, C> {
        Nullspace::new(builder.src)
    }
}

impl<T, C> Buildable<(Matrix<T, C>, Matrix<T, C>)> for QRDecomposition<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    Matrix<T, C>: PivotSolvable<T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
{
    fn build(builder: Self) -> (Matrix<T, C>, Matrix<T, C>) {
        (builder.q, builder.r)
    }
}

impl<T, C> Buildable<Matrix<T, C>> for Nullspace<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    Matrix<T, C>: PivotSolvable<T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
    T: AddAssign<T>,
{
    fn build(builder: Self) -> Matrix<T, C> {
        solve_nullspace_basis(&builder.r, &builder.free_vars)
    }
}

impl<T, C> Echelonizable for QRDecomposition<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
{
    fn has_element_at(&self, row: usize, k: usize) -> bool {
        row < self.r.rows && k < self.r.cols
    }

    fn pivot_row(&self, edge_row: usize, k: usize) -> Option<usize> {
        self.r.pivot_row(edge_row, k)
    }

    fn reduce_down(&mut self, edge_row: usize, k: usize) {
        let quots = self.r.reduce_rows_below(edge_row, k);
        self.q.add_row_block_below_with_weights(&quots);
        #[cfg(feature = "verify-matrices")]
        {
            assert_eq!(self.invariant, &(&self.q.transpose() * &self.r));
        }
    }

    fn advance(&mut self, edge_row: usize, k: usize) -> usize {
        if self.r.get(edge_row, k).is_zero() {
            edge_row
        } else {
            let quots = self.r.reduce_rows_above(edge_row, k);
            self.q.add_row_block_above_with_weights(&quots);
            #[cfg(feature = "verify-matrices")]
            {
                assert_eq!(self.invariant, &(&self.q.transpose() * &self.r));
            }

            edge_row + 1
        }
    }

    fn swap(&mut self, edge_row: usize, pivot_row: usize) {
        self.r.swap_rows(edge_row, pivot_row);
        self.q.swap_rows(edge_row, pivot_row);
        #[cfg(feature = "verify-matrices")]
        {
            assert_eq!(self.invariant, &(&self.q.transpose() * &self.r));
        }
    }

    fn finalize(&mut self) {
        let temp = self.q.transpose();
        #[cfg(feature = "verify-matrices")]
        {
            assert_eq!(self.invariant, &(&temp * &self.r));
        }
        self.q = temp;
    }
}

impl<T, C> Echelonizable for Nullspace<T, C>
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
{
    fn has_element_at(&self, row: usize, k: usize) -> bool {
        row < self.r.rows && k < self.r.cols
    }

    fn pivot_row(&self, edge_row: usize, k: usize) -> Option<usize> {
        self.r.pivot_row(edge_row, k)
    }

    fn reduce_down(&mut self, edge_row: usize, k: usize) {
        self.r.reduce_rows_below(edge_row, k);
    }

    fn advance(&mut self, edge_row: usize, k: usize) -> usize {
        if self.r.get(edge_row, k).is_zero() {
            self.free_vars.push(k);

            edge_row
        } else {
            self.r.reduce_rows_above(edge_row, k);

            edge_row + 1
        }
    }

    fn swap(&mut self, edge_row: usize, pivot_row: usize) {
        self.r.swap_rows(edge_row, pivot_row);
    }

    fn finalize(&mut self) {
        let nullcount = self.r.null_bottom_row_count();

        // Pivots = row count - null bottom row count
        // Free + pivots = column count
        let n_free = self.free_vars.len();
        let n_rows = self.r.rows;
        let n_cols = self.r.cols;
        if n_free + n_rows < n_cols + nullcount {
            let missing = n_cols + nullcount - n_rows - n_free;
            for i in (n_cols - missing)..n_cols {
                self.free_vars.push(i);
            }
        }
    }
}

/// Function to compute the unnormalized row echelon form of a matrix
#[allow(unused)]
pub fn unnormalized_row_echelon_form_builder<T, C, R, B>(matrix_a: B) -> R
where
    T: RingElement + DebugIfVerify,
    C: LinearContainer<T> + DebugIfVerify,
    R: Echelonizable,
    B: Buildable<R>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
{
    let mut result: R = B::build(matrix_a);

    let mut k = 0;
    let mut edge_row = 0;
    while result.has_element_at(edge_row, k) {
        let Some(pivot_row) = result.pivot_row(edge_row, k) else {
            edge_row = result.advance(edge_row, k);
            k += 1; // Column is ready, all zeroes below diagonal
            continue;
        };

        if pivot_row != edge_row {
            result.swap(edge_row, pivot_row);
        }

        // Reduce rows below the edge row
        result.reduce_down(edge_row, k);

        // Note: Do not increment k here, column is not ready
    }

    result.finalize();

    result
}

#[cfg(test)]
mod tests {
    use num::{BigInt, BigRational, Zero};

    use super::*;

    #[test]
    fn test_happy_path() {
        #[rustfmt::skip]
        let original = Matrix::new(4, 4, vec![
         1i64, 0, 0, 2,
            0, 0, 0, 3,
            0, 0, 0, 5,
            0, 0, 0, 6,
        ]);

        let builder = QRDecompositionBuilder::new(original.clone());
        let decomp = unnormalized_row_echelon_form_builder::<
            i64,
            Vec<i64>,
            QRDecomposition<i64, Vec<i64>>,
            QRDecompositionBuilder<i64, Vec<i64>>,
        >(builder);

        let (q, r) = QRDecomposition::<i64, Vec<i64>>::build(decomp);

        let k = r.null_bottom_row_count();
        assert_eq!(k, 2);

        #[rustfmt::skip]
        let expected = Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
               0, 0, 0, 1,
               0, 0, 0, 0,
               0, 0, 0, 0,
           ]);
        assert_eq!(r.data, expected.data);

        #[rustfmt::skip]
        let transform =
        &Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
               0, 1, 0, 0,
               0, 1, 1, 0,
               0, 2, 0, 1,
           ]) * &(
        &Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
               0, 0, 1, 0,
               0, 1, 0, 0,
               0, 0, 0, 1,
            ]) * &(
        &Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
                0, 1, 0, 0,
                0, 1, 1, 0,
                0, 0, 0, 1,
            ]) * &(
        &Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
                0, 0, 1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1,
        ]) * &(
        &Matrix::new(4, 4, vec![
            1i64, 0, 0, 0,
               0, 1, 0, 0,
               0, 2, 1, 0,
               0, 0, 0, 1,
        ]) *
        &Matrix::new(4, 4, vec![
            1i64, 2, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1,
           ])))));
        assert_eq!(q.data, transform.data);
    }

    #[test]
    fn test_nullspace_solver() {
        let to_big_rat = |elem| BigRational::from((BigInt::from(elem), BigInt::from(1)));
        #[rustfmt::skip]
        let original = Matrix::<BigRational, Vec<BigRational>>::new(3, 5, vec![
            1, 2,  3,  5,  7,
            0, 0, 17, 13, 11,
            0, 0, 0, 0, 0
        ].into_iter().map(to_big_rat).collect());
        // TODO: Does row echelon form solver return the same matrix if no further changes are needed?
        // Do we still get the right free variables?
        let free_vars = vec![1usize, 3, 4];

        let res = solve_nullspace_basis(&original, &free_vars);

        let m = original.rows;
        let k = res.rows;
        let n = res.cols;
        assert_eq!(n, original.cols);
        for i in 0..k {
            let start = i * n;
            for row in 0..m {
                let mut accum = BigRational::zero();
                for j in 0..n {
                    assert!(start + j < res.data.len());
                    std::ops::AddAssign::add_assign(
                        &mut accum,
                        &(res.get(i, j) * original.get(row, j)),
                    );
                }
                assert_eq!(BigRational::zero(), accum, "{} {}", i, row);
            }
        }
    }
}
