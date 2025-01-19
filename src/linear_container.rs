use std::ops::Mul;

pub trait LinearContainer<T: Clone + PartialEq>: Clone + PartialEq {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> &T;
    fn get_mut(&mut self, index: usize) -> &mut T;

    /// Create a new container with all elements initialized to the given value
    fn with_size(initial_value: T, size: usize) -> Self;

    fn add_assign(&mut self, idx1_start: usize, idx2_start: usize, length: usize, scalar: &T)
    where
        T: crate::ring_element::RingElement + Clone,
        for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    {
        for offset in 0..length {
            let idx2 = idx2_start + offset;
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx2 < self.len());
            }
            let delta = self.get(idx2) * scalar;
            let idx1 = idx1_start + offset;
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx1 < self.len());
            }
            self.get_mut(idx1).add_assign(&delta);
        }
    }

    fn swap(&mut self, idx1_start: usize, idx2_start: usize, length: usize)
    where
        T: Clone,
    {
        let temp: Vec<T> = (0..length)
            .map(|offset| {
                #[cfg(feature = "verify-indexes")]
                {
                    assert!(idx1_start + offset < self.len());
                }
                self.get(idx1_start + offset).clone()
            })
            .collect();

        for offset in 0..length {
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx2_start + offset < self.len());
            }
            let value = self.get(idx2_start + offset).clone();
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx1_start + offset < self.len());
            }
            *self.get_mut(idx1_start + offset) = value;
        }

        for (offset, value) in temp.into_iter().enumerate() {
            #[cfg(feature = "verify-indexes")]
            {
                assert!(idx2_start + offset < self.len());
            }
            *self.get_mut(idx2_start + offset) = value;
        }
    }
}

impl<T: Clone + PartialEq> LinearContainer<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> &T {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(index < self.len());
        }
        &self[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        #[cfg(feature = "verify-indexes")]
        {
            assert!(index < self.len());
        }
        &mut self[index]
    }

    fn with_size(initial_value: T, size: usize) -> Self {
        vec![initial_value; size]
    }
}
