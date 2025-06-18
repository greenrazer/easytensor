use std::{
    collections::HashMap,
    ops::{Index, IndexMut, RangeInclusive},
};

use num_traits::Zero;

#[derive(Debug, Clone, PartialEq)]
struct TensorShape {
    shape: Vec<usize>,
    strides: Vec<usize>,
    linear_offset: usize,
}

impl TensorShape {
    fn new(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        TensorShape {
            shape,
            strides,
            linear_offset: 0,
        }
    }

    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn ravel_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Indices length does not match tensor shape dimensions.");
        }

        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    fn unravel_index(&self, index: usize) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remaining_index = index;

        for (i, &stride) in self.strides.iter().enumerate() {
            indices[i] = remaining_index / stride;
            remaining_index %= stride;
        }

        indices
    }

    fn permute(&self, permuted_indices: &[usize]) -> Self {
        let shape = permuted_indices.iter().map(|&i| self.shape[i]).collect();
        let strides = permuted_indices.iter().map(|&i| self.strides[i]).collect();
        Self {
            shape,
            strides,
            linear_offset: self.linear_offset,
        }
    }

    fn merge(&self, dim_ranges: &[RangeInclusive<usize>]) -> Self {
        // Sort ranges by start index
        let mut sorted_dim_ranges: Vec<_> = dim_ranges.iter().collect();
        sorted_dim_ranges.sort_by_key(|r| r.start());

        // Check that the ranges don't overlap
        if !sorted_dim_ranges
            .windows(2)
            .all(|w| w[0].end() < w[1].start())
        {
            panic!("Ranges must not overlap");
        }

        // Compute the final number of dimensions
        let mut final_num_dimensions = self.shape.len();
        for range in sorted_dim_ranges.iter() {
            final_num_dimensions -= range.end() - range.start();
        }

        let mut shape = Vec::with_capacity(final_num_dimensions);
        let mut strides = Vec::with_capacity(final_num_dimensions);
        let mut current_pos = 0;

        for range in sorted_dim_ranges {
            // Add dimensions before this range
            shape.extend_from_slice(&self.shape[current_pos..*range.start()]);
            strides.extend_from_slice(&self.strides[current_pos..*range.start()]);

            // Calculate merged dimension size
            let merged_size: usize = self.shape[range.clone()].iter().product();
            shape.push(merged_size);
            let stride_size: usize = self.strides[range.end().clone()];
            strides.push(stride_size);

            current_pos = range.end() + 1;
        }

        // Add remaining dimensions after the last range
        shape.extend_from_slice(&self.shape[current_pos..]);
        strides.extend_from_slice(&self.strides[current_pos..]);

        Self {
            shape,
            strides,
            linear_offset: self.linear_offset,
        }
    }

    fn split(&self, splits: &HashMap<usize, Vec<usize>>) -> Self {
        if splits.is_empty() {
            return self.clone();
        }

        // Collect and sort dimension indices to process them in order
        let mut sorted_dims: Vec<_> = splits.keys().cloned().collect();
        sorted_dims.sort();

        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut current_dim = 0;

        for &split_dim in &sorted_dims {
            // Add dimensions before this split
            shape.extend_from_slice(&self.shape[current_dim..split_dim]);
            strides.extend_from_slice(&self.strides[current_dim..split_dim]);

            // Process the split for this dimension
            let split_sizes = &splits[&split_dim];
            let original_size = self.shape[split_dim];
            let original_stride = self.strides[split_dim];

            // Calculate the product of non-zero sizes and count zeros
            let mut non_zero_product = 1usize;
            let mut zero_index = None;

            for (i, &size) in split_sizes.iter().enumerate() {
                if size == 0 {
                    if zero_index.is_some() {
                        panic!("Cannot have more than one wildcard (0) in split sizes");
                    }
                    zero_index = Some(i);
                } else {
                    non_zero_product = non_zero_product * size;
                }
            }

            // Create the new sizes, inferring wildcards
            let mut final_sizes = split_sizes.clone();
            if let Some(zero_index) = zero_index {
                if original_size % non_zero_product != 0 {
                    panic!(
                        "Cannot split dimension of size {} into sizes {:?} - not evenly divisible",
                        original_size, split_sizes
                    );
                }
                let inferred_size = original_size / non_zero_product;
                final_sizes[zero_index] = inferred_size;
            }

            // Calculate strides for the split dimensions
            // The stride decreases as we go through the split dimensions
            let mut current_stride = original_stride;
            for &size in final_sizes.iter().rev() {
                strides.push(current_stride);
                current_stride = current_stride * size;
            }

            // Reverse the strides we just added to maintain correct order
            let start_idx = strides.len() - final_sizes.len();
            strides[start_idx..].reverse();

            // Add the split dimensions to shape
            shape.extend_from_slice(&final_sizes);

            current_dim = split_dim + 1;
        }

        // Add remaining dimensions after the last split
        shape.extend_from_slice(&self.shape[current_dim..]);
        strides.extend_from_slice(&self.strides[current_dim..]);

        Self {
            shape,
            strides,
            linear_offset: self.linear_offset,
        }
    }

    fn slice(&self, slices: &HashMap<usize, RangeInclusive<usize>>) -> Self {
        if slices.is_empty() {
            return self.clone();
        }

        let mut new_shape = self.shape.clone();
        let mut additional_offset = 0;

        for (&dim, range) in slices {
            let start = *range.start();
            let end = *range.end();

            // Calculate offset contribution from this dimension
            additional_offset += start * self.strides[dim];

            // Update shape for this dimension (inclusive range)
            new_shape[dim] = end - start + 1;
        }

        Self {
            shape: new_shape,
            strides: self.strides.clone(),
            linear_offset: self.linear_offset + additional_offset,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T: Zero + Clone> TensorStorage<T> {
    fn zeros(size: usize) -> Self {
        TensorStorage {
            data: vec![T::zero(); size],
        }
    }
}

impl<T> Index<usize> for TensorStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for TensorStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T: Zero + Clone> Tensor<T> {
    fn zeros(shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage::<T>::zeros(shape.size());
        Tensor { shape, storage }
    }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.storage[self.shape.ravel_index(indices)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.storage[self.shape.ravel_index(indices)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros_basic() {
        let tensor = Tensor::<f32>::zeros(vec![2, 3]);

        assert_eq!(tensor.shape.shape, vec![2, 3]);

        assert_eq!(tensor.storage.data.len(), 6);

        assert!(tensor.storage.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ravel_index() {
        let shape = TensorShape::new(vec![5]);

        assert_eq!(shape.ravel_index(&[0]), 0);
        assert_eq!(shape.ravel_index(&[1]), 1);
        assert_eq!(shape.ravel_index(&[2]), 2);
        assert_eq!(shape.ravel_index(&[3]), 3);
        assert_eq!(shape.ravel_index(&[4]), 4);

        let shape = TensorShape::new(vec![2, 3]);

        assert_eq!(shape.ravel_index(&[0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 2]), 2);
        assert_eq!(shape.ravel_index(&[1, 0]), 3);
        assert_eq!(shape.ravel_index(&[1, 1]), 4);
        assert_eq!(shape.ravel_index(&[1, 2]), 5);

        let shape = TensorShape::new(vec![2, 3, 4]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 2]), 2);
        assert_eq!(shape.ravel_index(&[0, 0, 3]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 4);
        assert_eq!(shape.ravel_index(&[0, 1, 1]), 5);
        assert_eq!(shape.ravel_index(&[0, 2, 3]), 11);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 12);
        assert_eq!(shape.ravel_index(&[1, 1, 1]), 17);
        assert_eq!(shape.ravel_index(&[1, 2, 3]), 23);

        let shape = TensorShape::new(vec![2, 2, 2, 2]);

        assert_eq!(shape.ravel_index(&[0, 0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 0]), 2);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 1]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 0, 0]), 4);
        assert_eq!(shape.ravel_index(&[1, 0, 0, 0]), 8);
        assert_eq!(shape.ravel_index(&[1, 1, 1, 1]), 15);

        let shape = TensorShape::new(vec![10, 20, 30]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 30);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 600);
        assert_eq!(shape.ravel_index(&[5, 10, 15]), 5 * 600 + 10 * 30 + 15);
        assert_eq!(shape.ravel_index(&[9, 19, 29]), 9 * 600 + 19 * 30 + 29);

        let shape = TensorShape::new(vec![1, 1, 1]);

        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);

        let shape = TensorShape::new(vec![3, 4]);

        let expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        let mut index = 0;
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(shape.ravel_index(&[i, j]), expected[index]);
                index += 1;
            }
        }
    }

    #[test]
    fn test_unravel_index() {
        let shape = TensorShape::new(vec![5]);

        assert_eq!(shape.unravel_index(0), vec![0]);
        assert_eq!(shape.unravel_index(1), vec![1]);
        assert_eq!(shape.unravel_index(2), vec![2]);
        assert_eq!(shape.unravel_index(3), vec![3]);
        assert_eq!(shape.unravel_index(4), vec![4]);

        let shape = TensorShape::new(vec![2, 3]);

        assert_eq!(shape.unravel_index(0), vec![0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 2]);
        assert_eq!(shape.unravel_index(3), vec![1, 0]);
        assert_eq!(shape.unravel_index(4), vec![1, 1]);
        assert_eq!(shape.unravel_index(5), vec![1, 2]);

        let shape = TensorShape::new(vec![2, 3, 4]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 0, 2]);
        assert_eq!(shape.unravel_index(3), vec![0, 0, 3]);
        assert_eq!(shape.unravel_index(4), vec![0, 1, 0]);
        assert_eq!(shape.unravel_index(5), vec![0, 1, 1]);
        assert_eq!(shape.unravel_index(11), vec![0, 2, 3]);
        assert_eq!(shape.unravel_index(12), vec![1, 0, 0]);
        assert_eq!(shape.unravel_index(17), vec![1, 1, 1]);
        assert_eq!(shape.unravel_index(23), vec![1, 2, 3]);

        let shape = TensorShape::new(vec![2, 2, 2, 2]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 0, 1]);
        assert_eq!(shape.unravel_index(2), vec![0, 0, 1, 0]);
        assert_eq!(shape.unravel_index(3), vec![0, 0, 1, 1]);
        assert_eq!(shape.unravel_index(4), vec![0, 1, 0, 0]);
        assert_eq!(shape.unravel_index(8), vec![1, 0, 0, 0]);
        assert_eq!(shape.unravel_index(15), vec![1, 1, 1, 1]);

        let shape = TensorShape::new(vec![10, 20, 30]);

        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(shape.unravel_index(1), vec![0, 0, 1]);
        assert_eq!(shape.unravel_index(30), vec![0, 1, 0]);
        assert_eq!(shape.unravel_index(600), vec![1, 0, 0]);
        assert_eq!(shape.unravel_index(5 * 600 + 10 * 30 + 15), vec![5, 10, 15]);
        assert_eq!(shape.unravel_index(9 * 600 + 19 * 30 + 29), vec![9, 19, 29]);

        let shape = TensorShape::new(vec![1, 1, 1]);
        assert_eq!(shape.unravel_index(0), vec![0, 0, 0]);

        let shape = TensorShape::new(vec![]);
        assert_eq!(shape.unravel_index(0), vec![]);

        let shape = TensorShape::new(vec![3, 4]);

        let expected_indices = [
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
            vec![1, 3],
            vec![2, 0],
            vec![2, 1],
            vec![2, 2],
            vec![2, 3],
        ];

        for (flat_index, expected_multi_index) in expected_indices.iter().enumerate() {
            assert_eq!(shape.unravel_index(flat_index), *expected_multi_index);
        }

        let shape = TensorShape::new(vec![4, 5, 6]);

        for flat_index in 0..(4 * 5 * 6) {
            let multi_index = shape.unravel_index(flat_index);
            let recovered_flat_index = shape.ravel_index(&multi_index);
            assert_eq!(flat_index, recovered_flat_index);
        }
    }

    #[test]
    fn test_tensor_index() {
        let mut tensor_1d = Tensor::<f32>::zeros(vec![5]);
        assert_eq!(tensor_1d[&[0]], 0.0);
        assert_eq!(tensor_1d[&[4]], 0.0);

        tensor_1d[&[0]] = 1.0;
        tensor_1d[&[1]] = 2.0;
        tensor_1d[&[4]] = 5.0;

        assert_eq!(tensor_1d[&[0]], 1.0);
        assert_eq!(tensor_1d[&[1]], 2.0);
        assert_eq!(tensor_1d[&[2]], 0.0);
        assert_eq!(tensor_1d[&[4]], 5.0);

        let mut tensor_2d = Tensor::<i32>::zeros(vec![3, 4]);
        assert_eq!(tensor_2d[&[0, 0]], 0);
        assert_eq!(tensor_2d[&[2, 3]], 0);

        tensor_2d[&[0, 0]] = 10;
        tensor_2d[&[0, 3]] = 13;
        tensor_2d[&[1, 2]] = 42;
        tensor_2d[&[2, 3]] = 99;

        assert_eq!(tensor_2d[&[0, 0]], 10);
        assert_eq!(tensor_2d[&[0, 3]], 13);
        assert_eq!(tensor_2d[&[1, 2]], 42);
        assert_eq!(tensor_2d[&[2, 3]], 99);
        assert_eq!(tensor_2d[&[0, 1]], 0);
        assert_eq!(tensor_2d[&[1, 0]], 0);

        let mut tensor_3d = Tensor::<f64>::zeros(vec![2, 3, 4]);
        tensor_3d[&[0, 0, 0]] = 1.1;
        tensor_3d[&[1, 2, 3]] = 2.2;
        tensor_3d[&[0, 1, 2]] = 3.3;
        tensor_3d[&[1, 0, 1]] = 4.4;

        assert_eq!(tensor_3d[&[0, 0, 0]], 1.1);
        assert_eq!(tensor_3d[&[1, 2, 3]], 2.2);
        assert_eq!(tensor_3d[&[0, 1, 2]], 3.3);
        assert_eq!(tensor_3d[&[1, 0, 1]], 4.4);
        assert_eq!(tensor_3d[&[0, 0, 1]], 0.0);
        assert_eq!(tensor_3d[&[1, 1, 1]], 0.0);

        let mut tensor_mut = Tensor::<i32>::zeros(vec![2, 2]);
        {
            let value_ref = &mut tensor_mut[&[0, 1]];
            *value_ref = 42;
        }
        assert_eq!(tensor_mut[&[0, 1]], 42);

        tensor_mut[&[1, 0]] += 10;
        tensor_mut[&[1, 0]] *= 2;
        assert_eq!(tensor_mut[&[1, 0]], 20);
    }

    #[test]
    fn test_tensor_storage_and_consistency() {
        let mut storage = TensorStorage::<u8>::zeros(5);
        assert_eq!(storage[0], 0);
        assert_eq!(storage[4], 0);

        storage[0] = 100;
        storage[2] = 200;
        storage[4] = 255;

        assert_eq!(storage[0], 100);
        assert_eq!(storage[1], 0);
        assert_eq!(storage[2], 200);
        assert_eq!(storage[3], 0);
        assert_eq!(storage[4], 255);

        let mut tensor = Tensor::<i16>::zeros(vec![3, 4]);
        let test_cases = vec![
            ([0, 0], 100),
            ([0, 3], 103),
            ([1, 1], 111),
            ([2, 0], 200),
            ([2, 3], 203),
        ];

        for &(indices, value) in &test_cases {
            tensor[&indices] = value;
        }

        for &(indices, expected_value) in &test_cases {
            assert_eq!(
                tensor[&indices], expected_value,
                "Failed for indices {:?}",
                indices
            );
        }

        for &(indices, expected_value) in &test_cases {
            let flat_index = tensor.shape.ravel_index(&indices);
            assert_eq!(
                tensor.storage[flat_index], expected_value,
                "Flat index consistency failed for indices {:?}",
                indices
            );
        }

        let mut tensor_3d = Tensor::<f32>::zeros(vec![2, 3, 4]);
        let mut expected_value = 1.0;

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = expected_value;
                    expected_value += 1.0;
                }
            }
        }

        expected_value = 1.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(
                        tensor_3d[&[i, j, k]],
                        expected_value,
                        "Failed at indices [{}, {}, {}]",
                        i,
                        j,
                        k
                    );
                    expected_value += 1.0;
                }
            }
        }
    }

    #[test]
    fn test_permute() {
        // Test 2D permutation (transpose)
        let shape_2d = TensorShape::new(vec![3, 4]);
        assert_eq!(shape_2d.shape, vec![3, 4]);
        assert_eq!(shape_2d.strides, vec![4, 1]);

        let shape_2d = shape_2d.permute(&[1, 0]); // transpose
        assert_eq!(shape_2d.shape, vec![4, 3]);
        assert_eq!(shape_2d.strides, vec![1, 4]);

        // Test 3D permutation
        let shape_3d = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape_3d.shape, vec![2, 3, 4]);
        assert_eq!(shape_3d.strides, vec![12, 4, 1]);

        // Permute to [2, 0, 1] - move last dimension to front
        let shape_3d = shape_3d.permute(&[2, 0, 1]);
        assert_eq!(shape_3d.shape, vec![4, 2, 3]);
        assert_eq!(shape_3d.strides, vec![1, 12, 4]);

        // Test 4D permutation
        let shape_4d = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.strides, vec![60, 20, 5, 1]);

        // Reverse the dimensions
        let shape_4d = shape_4d.permute(&[3, 2, 1, 0]);
        assert_eq!(shape_4d.shape, vec![5, 4, 3, 2]);
        assert_eq!(shape_4d.strides, vec![1, 5, 20, 60]);

        // Test identity permutation (no change)
        let shape_identity = TensorShape::new(vec![2, 3, 4]);
        let original_shape = shape_identity.shape.clone();
        let original_strides = shape_identity.strides.clone();

        let shape_identity = shape_identity.permute(&[0, 1, 2]);
        assert_eq!(shape_identity.shape, original_shape);
        assert_eq!(shape_identity.strides, original_strides);

        // Test with single dimension
        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let shape_1d = shape_1d.permute(&[0]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        // Test permutation preserves index mapping
        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let permuted_shape = original_shape.permute(&[1, 2, 0]);

        // Verify that the same multi-dimensional indices map correctly
        // Original: [i, j, k] -> permuted: [j, k, i]
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let original_flat = original_shape.ravel_index(&[i, j, k]);
                    let permuted_flat = permuted_shape.ravel_index(&[j, k, i]);
                    assert_eq!(
                        original_flat, permuted_flat,
                        "Index mismatch for [{}, {}, {}] vs [{}, {}, {}]",
                        i, j, k, j, k, i
                    );
                }
            }
        }

        // Test empty shape
        let empty_shape = TensorShape::new(vec![]).permute(&[]);
        assert_eq!(empty_shape.shape, vec![]);
        assert_eq!(empty_shape.strides, vec![]);
    }

    #[test]
    fn test_merge() {
        // Test single range merge in the middle
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape.strides, vec![60, 20, 5, 1]);

        let merged_shape = shape.merge(&[1..=2]);
        assert_eq!(merged_shape.shape, vec![2, 12, 5]); // 3 * 4 = 12
        assert_eq!(merged_shape.strides, vec![60, 5, 1]);

        // Test merge at the beginning
        let shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = shape.merge(&[0..=1]);
        assert_eq!(merged_shape.shape, vec![6, 4]); // 2 * 3 = 6
        assert_eq!(merged_shape.strides, vec![4, 1]);

        // Test merge at the end
        let shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = shape.merge(&[1..=2]);
        assert_eq!(merged_shape.shape, vec![2, 12]); // 3 * 4 = 12
        assert_eq!(merged_shape.strides, vec![12, 1]);

        // Test merging entire tensor to single dimension
        let shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = shape.merge(&[0..=2]);
        assert_eq!(merged_shape.shape, vec![24]); // 2 * 3 * 4 = 24
        assert_eq!(merged_shape.strides, vec![1]);

        // Test multiple non-overlapping ranges
        let shape = TensorShape::new(vec![2, 3, 4, 5, 6]);
        let merged_shape = shape.merge(&[0..=1, 3..=4]);
        assert_eq!(merged_shape.shape, vec![6, 4, 30]); // [2*3, 4, 5*6] = [6, 4, 30]
        assert_eq!(merged_shape.strides, vec![120, 30, 1]);

        // Test single dimension ranges (no actual merging)
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        let merged_shape = shape.merge(&[1..=1, 3..=3]);
        assert_eq!(merged_shape.shape, vec![2, 3, 4, 5]); // No change
        assert_eq!(merged_shape.strides, vec![60, 20, 5, 1]);

        // Test empty ranges (no operation)
        let shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = shape.merge(&[]);
        assert_eq!(merged_shape.shape, vec![2, 3, 4]);
        assert_eq!(merged_shape.strides, vec![12, 4, 1]);

        // Test complex case with multiple ranges
        let shape = TensorShape::new(vec![2, 3, 4, 5, 6, 7]);
        let merged_shape = shape.merge(&[0..=0, 2..=3, 5..=5]);
        assert_eq!(merged_shape.shape, vec![2, 3, 20, 6, 7]); // [2, 3, 4*5, 6, 7] = [2, 3, 20, 6, 7]
        assert_eq!(merged_shape.strides, vec![2520, 840, 42, 7, 1]);

        // Test that index mapping is preserved after merging
        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = original_shape.merge(&[1..=2]);

        // Verify some key mappings: [i, j, k] in original -> [i, j*4+k] in merged
        assert_eq!(
            original_shape.ravel_index(&[0, 0, 0]),
            merged_shape.ravel_index(&[0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[0, 1, 2]),
            merged_shape.ravel_index(&[0, 6])
        ); // j=1, k=2 -> 1*4+2=6
        assert_eq!(
            original_shape.ravel_index(&[1, 2, 3]),
            merged_shape.ravel_index(&[1, 11])
        ); // j=2, k=3 -> 2*4+3=11

        // Test with 1D tensor
        let shape = TensorShape::new(vec![10]);
        let merged_shape = shape.merge(&[0..=0]);
        assert_eq!(merged_shape.shape, vec![10]);
        assert_eq!(merged_shape.strides, vec![1]);

        // Test larger merging scenario
        let shape = TensorShape::new(vec![2, 3, 4, 5, 6, 7, 8]);
        let merged_shape = shape.merge(&[1..=3, 5..=6]);
        assert_eq!(merged_shape.shape, vec![2, 60, 6, 56]); // [2, 3*4*5, 6, 7*8] = [2, 60, 6, 56]
        assert_eq!(merged_shape.strides, vec![20160, 336, 56, 1]);
    }

    #[test]
    fn test_split() {
        // Test basic split without wildcards
        let shape = TensorShape::new(vec![2, 12, 5]);
        let mut splits = HashMap::new();
        splits.insert(1, vec![3, 4]); // Split dimension 1 (size 12) into [3, 4]
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(split_shape.strides, vec![60, 20, 5, 1]);

        // Test split with wildcard (zero)
        let shape = TensorShape::new(vec![24]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![2, 3, 0]); // Split dimension 0 (size 24) into [2, 3, ?] where ? = 24/(2*3) = 4
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 4, 1]);

        // Test multiple splits
        let shape = TensorShape::new(vec![6, 20, 7]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![2, 3]); // Split dimension 0 (size 6) into [2, 3]
        splits.insert(1, vec![4, 5]); // Split dimension 1 (size 20) into [4, 5]
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 5, 7]);
        assert_eq!(split_shape.strides, vec![420, 140, 35, 7, 1]);

        // Test split with wildcard in middle
        let shape = TensorShape::new(vec![2, 60, 3]);
        let mut splits = HashMap::new();
        splits.insert(1, vec![4, 0, 5]); // Split dimension 1 (size 60) into [4, ?, 5] where ? = 60/(4*5) = 3
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 4, 3, 5, 3]);
        assert_eq!(split_shape.strides, vec![180, 45, 15, 3, 1]);

        // Test split at the end
        let shape = TensorShape::new(vec![2, 3, 24]);
        let mut splits = HashMap::new();
        splits.insert(2, vec![4, 6]); // Split last dimension (size 24) into [4, 6]
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 6]);
        assert_eq!(split_shape.strides, vec![72, 24, 6, 1]);

        // Test split at the beginning
        let shape = TensorShape::new(vec![12, 3, 4]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![3, 4]); // Split first dimension (size 12) into [3, 4]
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![3, 4, 3, 4]);
        assert_eq!(split_shape.strides, vec![48, 12, 4, 1]);

        // Test no splits (empty HashMap)
        let shape = TensorShape::new(vec![2, 3, 4]);
        let split_shape = shape.split(&HashMap::new());
        assert_eq!(split_shape.shape, vec![2, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 4, 1]);

        // Test single dimension split
        let shape = TensorShape::new(vec![30]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![5, 6]);
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![5, 6]);
        assert_eq!(split_shape.strides, vec![6, 1]);

        // Test complex split with multiple wildcards in different dimensions
        let shape = TensorShape::new(vec![2, 60, 3, 40]);
        let mut splits = HashMap::new();
        splits.insert(1, vec![0, 5]); // Split dimension 1 (size 60) into [?, 5] where ? = 12
        splits.insert(3, vec![8, 0]); // Split dimension 3 (size 40) into [8, ?] where ? = 5
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 12, 5, 3, 8, 5]);
        assert_eq!(split_shape.strides, vec![7200, 600, 120, 40, 5, 1]);

        // Test that index mapping is preserved after splitting
        let original_shape = TensorShape::new(vec![6, 8]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![2, 3]); // Split first dimension 6 into [2, 3]
        splits.insert(1, vec![4, 2]); // Split second dimension 8 into [4, 2]
        let split_shape = original_shape.split(&splits);

        // Verify some key mappings: [i, j] in original -> [i/3, i%3, j/2, j%2] in split
        assert_eq!(
            original_shape.ravel_index(&[0, 0]),
            split_shape.ravel_index(&[0, 0, 0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[2, 6]),
            split_shape.ravel_index(&[0, 2, 3, 0])
        ); // i=2 -> [0, 2], j=6 -> [3, 0]
        assert_eq!(
            original_shape.ravel_index(&[5, 7]),
            split_shape.ravel_index(&[1, 2, 3, 1])
        ); // i=5 -> [1, 2], j=7 -> [3, 1]

        // Test edge case: split into single elements
        let shape = TensorShape::new(vec![4]);
        let mut splits = HashMap::new();
        splits.insert(0, vec![4, 1]);
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![4, 1]);
        assert_eq!(split_shape.strides, vec![1, 1]);

        // Test split that results in same total size
        let shape = TensorShape::new(vec![2, 3, 4]);
        let mut splits = HashMap::new();
        splits.insert(1, vec![1, 3]); // Split dimension 1 (size 3) into [1, 3]
        let split_shape = shape.split(&splits);
        assert_eq!(split_shape.shape, vec![2, 1, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 12, 4, 1]);
    }

    #[test]
    fn test_slice() {
        // Test basic 2D slicing
        let shape = TensorShape::new(vec![5, 6]);
        assert_eq!(shape.shape, vec![5, 6]);
        assert_eq!(shape.strides, vec![6, 1]);
        assert_eq!(shape.linear_offset, 0);

        let mut slices = HashMap::new();
        slices.insert(0, 1..=3); // Take rows 1, 2, 3
        slices.insert(1, 2..=4); // Take cols 2, 3, 4
        let sliced_shape = shape.slice(&slices);

        assert_eq!(sliced_shape.shape, vec![3, 3]); // 3 rows, 3 cols
        assert_eq!(sliced_shape.strides, vec![6, 1]); // Strides unchanged
        assert_eq!(sliced_shape.linear_offset, 1 * 6 + 2 * 1); // 1*6 + 2*1 = 8

        // Test 3D slicing
        let shape_3d = TensorShape::new(vec![4, 5, 6]);
        assert_eq!(shape_3d.strides, vec![30, 6, 1]);

        let mut slices_3d = HashMap::new();
        slices_3d.insert(0, 1..=2); // Take planes 1, 2
        slices_3d.insert(2, 1..=4); // Take last dim elements 1, 2, 3, 4
        let sliced_3d = shape_3d.slice(&slices_3d);

        assert_eq!(sliced_3d.shape, vec![2, 5, 4]); // [2 planes, 5 rows, 4 cols]
        assert_eq!(sliced_3d.strides, vec![30, 6, 1]); // Strides unchanged
        assert_eq!(sliced_3d.linear_offset, 1 * 30 + 1 * 1); // 1*30 + 0*6 + 1*1 = 31

        // Test single dimension slicing
        let shape_1d = TensorShape::new(vec![10]);
        let mut slices_1d = HashMap::new();
        slices_1d.insert(0, 3..=7); // Take elements 3, 4, 5, 6, 7
        let sliced_1d = shape_1d.slice(&slices_1d);

        assert_eq!(sliced_1d.shape, vec![5]); // 5 elements
        assert_eq!(sliced_1d.strides, vec![1]);
        assert_eq!(sliced_1d.linear_offset, 3); // Start at element 3

        // Test partial slicing (only some dimensions)
        let shape_partial = TensorShape::new(vec![3, 4, 5]);
        let mut slices_partial = HashMap::new();
        slices_partial.insert(1, 1..=2); // Only slice middle dimension
        let sliced_partial = shape_partial.slice(&slices_partial);

        assert_eq!(sliced_partial.shape, vec![3, 2, 5]); // Only middle dim changed
        assert_eq!(sliced_partial.strides, vec![20, 5, 1]);
        assert_eq!(sliced_partial.linear_offset, 1 * 5); // 0*20 + 1*5 + 0*1 = 5

        // Test empty slices (no operation)
        let shape_empty = TensorShape::new(vec![3, 4]);
        let sliced_empty = shape_empty.slice(&HashMap::new());
        assert_eq!(sliced_empty.shape, vec![3, 4]);
        assert_eq!(sliced_empty.strides, vec![4, 1]);
        assert_eq!(sliced_empty.linear_offset, 0);

        // Test single element slices
        let shape_single = TensorShape::new(vec![5, 5]);
        let mut slices_single = HashMap::new();
        slices_single.insert(0, 2..=2); // Single row
        slices_single.insert(1, 3..=3); // Single column
        let sliced_single = shape_single.slice(&slices_single);

        assert_eq!(sliced_single.shape, vec![1, 1]); // Single element
        assert_eq!(sliced_single.strides, vec![5, 1]);
        assert_eq!(sliced_single.linear_offset, 2 * 5 + 3 * 1); // 13

        // Test that index mapping is preserved after slicing
        let original_shape = TensorShape::new(vec![4, 6]);
        let mut test_slices = HashMap::new();
        test_slices.insert(0, 1..=2); // Rows 1, 2
        test_slices.insert(1, 2..=4); // Cols 2, 3, 4
        let sliced_test = original_shape.slice(&test_slices);

        // Verify that [0, 0] in sliced corresponds to [1, 2] in original
        let sliced_flat = sliced_test.linear_offset + sliced_test.ravel_index(&[0, 0]);
        let original_flat = original_shape.ravel_index(&[1, 2]);
        assert_eq!(sliced_flat, original_flat);

        // Verify that [1, 2] in sliced corresponds to [2, 4] in original
        let sliced_flat = sliced_test.linear_offset + sliced_test.ravel_index(&[1, 2]);
        let original_flat = original_shape.ravel_index(&[2, 4]);
        assert_eq!(sliced_flat, original_flat);
    }
}
