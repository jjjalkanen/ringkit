use std::cmp::Ord;
use std::ops::{Add, Mul};

use num::{BigInt, BigRational, One, Signed, Zero};

pub trait RingElement:
    Sized
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Clone
    + Eq
    + PartialEq
    + Zero
    + One
    + Signed
    + Ord
{
    fn quot(lhs: &Self, rhs: &Self) -> Self; // Truncated integer division
    fn add_assign(&mut self, other: &Self); // In-place addition
}

impl RingElement for i64 {
    fn quot(lhs: &Self, rhs: &Self) -> Self {
        lhs / rhs
    }

    fn add_assign(&mut self, other: &Self) {
        match self.checked_add(*other) {
            Some(value) => {
                *self = value;
            }
            None => {
                panic!("Overflow");
            }
        }
    }
}

impl RingElement for BigInt {
    fn quot(lhs: &Self, rhs: &Self) -> Self {
        lhs / rhs
    }

    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }
}

impl RingElement for BigRational {
    fn quot(lhs: &Self, rhs: &Self) -> Self {
        lhs / rhs
    }

    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }
}
