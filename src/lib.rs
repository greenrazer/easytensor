use std::ops::{Index, IndexMut, RangeInclusive};

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

    fn is_scalar(&self) -> bool {
        self.shape.is_empty()
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

    fn merge(&self, dim_range: RangeInclusive<usize>) -> Self {
        let (start, end) = (*dim_range.start(), *dim_range.end());

        assert!(
            start <= end && end < self.shape.len(),
            "Invalid dimension range for merge"
        );

        let merged_size = self.shape[dim_range.clone()].iter().product();
        let merged_stride = self.strides[end];

        let mut new_shape = Vec::with_capacity(self.shape.len() - (end - start));
        let mut new_strides = Vec::with_capacity(self.strides.len() - (end - start));

        new_shape.extend_from_slice(&self.shape[..start]);
        new_shape.push(merged_size);
        new_shape.extend_from_slice(&self.shape[end + 1..]);

        new_strides.extend_from_slice(&self.strides[..start]);
        new_strides.push(merged_stride);
        new_strides.extend_from_slice(&self.strides[end + 1..]);

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }

    fn split(&self, dim: usize, shape: &[usize]) -> Self {
        if dim >= self.shape.len() {
            panic!("Dimension index out of bounds");
        }

        let original_size = self.shape[dim];
        let original_stride = self.strides[dim];

        // Calculate the product of non-zero sizes and find wildcard
        let mut non_zero_product = 1usize;
        let mut zero_index = None;

        for (i, &size) in shape.iter().enumerate() {
            if size == 0 {
                if zero_index.is_some() {
                    panic!("Cannot have more than one wildcard (0) in split sizes");
                }
                zero_index = Some(i);
            } else {
                non_zero_product *= size;
            }
        }

        // Create the final sizes, inferring wildcards
        let mut final_sizes = shape.to_vec();
        if let Some(zero_index) = zero_index {
            if original_size % non_zero_product != 0 {
                panic!(
                    "Cannot split dimension of size {} into sizes {:?}",
                    original_size, shape
                );
            }
            let inferred_size = original_size / non_zero_product;
            final_sizes[zero_index] = inferred_size;
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        // Add dimensions before the split
        new_shape.extend_from_slice(&self.shape[..dim]);
        new_strides.extend_from_slice(&self.strides[..dim]);

        // Calculate strides for the split dimensions
        let mut current_stride = original_stride;
        for &size in final_sizes.iter().rev() {
            new_strides.push(current_stride);
            current_stride *= size;
        }

        // Reverse the strides we just added to maintain correct order
        let start_idx = new_strides.len() - final_sizes.len();
        new_strides[start_idx..].reverse();

        // Add the split dimensions to shape
        new_shape.extend_from_slice(&final_sizes);

        // Add remaining dimensions after the split
        if dim + 1 < self.shape.len() {
            new_shape.extend_from_slice(&self.shape[dim + 1..]);
            new_strides.extend_from_slice(&self.strides[dim + 1..]);
        }

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }

    fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        if dim >= self.shape.len() {
            panic!("Dimension index out of bounds");
        }

        let start = *range.start();
        let end = *range.end();

        if start > end || end >= self.shape[dim] {
            panic!("Invalid slice range for dimension {}", dim);
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start + 1; // inclusive range

        let additional_offset = start * self.strides[dim];

        Self {
            shape: new_shape,
            strides: self.strides.clone(),
            linear_offset: self.linear_offset + additional_offset,
        }
    }

    fn skip(&self, dim: usize, step: usize) -> Self {
        // perform the equivalent of slicing with no range, but a step
        if dim >= self.shape.len() {
            panic!("Dimension index out of bounds");
        }

        let mut new_strides = self.strides.clone();
        new_strides[dim] = new_strides[dim] * step;

        let mut new_shape = self.shape.clone();
        new_shape[dim] = new_shape[dim].div_ceil(step);

        Self {
            shape: new_shape,
            strides: new_strides,
            linear_offset: self.linear_offset,
        }
    }
}

impl From<&[usize]> for TensorShape {
    fn from(shape: &[usize]) -> Self {
        TensorShape::new(shape.to_vec())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> TensorStorage<T> {
    fn map<F, U>(&self, f: F) -> TensorStorage<U>
    where
        F: Fn(&T) -> U,
    {
        TensorStorage {
            data: self.data.iter().map(f).collect(),
        }
    }
}

impl<T: Clone> TensorStorage<T> {
    fn reduce_all<F>(&self, f: F) -> TensorStorage<T>
    where
        F: Fn(&T, &T) -> T,
        T: Clone,
    {
        let v = self
            .data
            .iter()
            .cloned()
            .reduce(|a, b| f(&a, &b))
            .expect("Cannot reduce an empty tensor storage");
        TensorStorage { data: vec![v] }
    }
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

impl<T> Tensor<T> {
    fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
    {
        Tensor {
            shape: self.shape.clone(),
            storage: self.storage.map(f),
        }
    }
}

impl<T: Clone> Tensor<T> {
    fn permute(&self, permuted_indices: &[usize]) -> Self {
        Tensor {
            shape: self.shape.permute(permuted_indices),
            storage: self.storage.clone(),
        }
    }

    fn merge(&self, dim_range: RangeInclusive<usize>) -> Self {
        Tensor {
            shape: self.shape.merge(dim_range),
            storage: self.storage.clone(),
        }
    }

    fn split(&self, dim: usize, shape: &[usize]) -> Self {
        Tensor {
            shape: self.shape.split(dim, shape),
            storage: self.storage.clone(),
        }
    }

    fn slice(&self, dim: usize, range: RangeInclusive<usize>) -> Self {
        Tensor {
            shape: self.shape.slice(dim, range),
            storage: self.storage.clone(),
        }
    }

    fn skip(&self, dim: usize, step: usize) -> Self {
        Tensor {
            shape: self.shape.skip(dim, step),
            storage: self.storage.clone(),
        }
    }
}

impl<T: Zero + Clone> Tensor<T> {
    fn zeros(shape: Vec<usize>) -> Self {
        let shape = TensorShape::new(shape);
        let storage = TensorStorage::<T>::zeros(shape.size());
        Tensor { shape, storage }
    }

    fn reduce<F>(&self, dim: usize, f: F) -> Tensor<T>
    where
        F: Fn(&T, &T) -> T,
    {
        if dim >= self.shape.shape.len() {
            panic!(
                "dim {} out of bounds for tensor with {} dimensions",
                dim,
                self.shape.shape.len()
            );
        }

        let mut result_shape = self.shape.shape.clone();
        result_shape.remove(dim);
        let result_tensor_shape: TensorShape = result_shape.as_slice().into();

        if result_tensor_shape.is_scalar() {
            return Tensor {
                shape: result_tensor_shape,
                storage: self.storage.reduce_all(f),
            };
        }

        let mut result_storage: TensorStorage<T> = TensorStorage::zeros(result_tensor_shape.size());

        // Initialize the result with the first slice along the reduction dimension
        let total_elements = self.shape.size();
        for flat_idx in 0..total_elements {
            let mut multi_idx = self.shape.unravel_index(flat_idx);
            
            // Only process elements where the reduction dimension index is 0
            if multi_idx[dim] == 0 {
                multi_idx.remove(dim);
                let result_flat_idx = result_tensor_shape.ravel_index(&multi_idx);
                result_storage[result_flat_idx] = self.storage[self.shape.linear_offset + flat_idx].clone();
            }
        }

        // Now reduce the remaining slices (dim index > 0) into the initialized result
        for flat_idx in 0..total_elements {
            let mut multi_idx = self.shape.unravel_index(flat_idx);
            
            // Only process elements where the reduction dimension index is > 0
            if multi_idx[dim] > 0 {
                multi_idx.remove(dim);
                let result_flat_idx = result_tensor_shape.ravel_index(&multi_idx);

                result_storage[result_flat_idx] = f(
                    &result_storage[result_flat_idx],
                    &self.storage[self.shape.linear_offset + flat_idx],
                );
            }
        }

        Tensor {
            shape: result_tensor_shape,
            storage: result_storage,
        }
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
        let shape_2d = TensorShape::new(vec![3, 4]);
        assert_eq!(shape_2d.shape, vec![3, 4]);
        assert_eq!(shape_2d.strides, vec![4, 1]);

        let shape_2d = shape_2d.permute(&[1, 0]);
        assert_eq!(shape_2d.shape, vec![4, 3]);
        assert_eq!(shape_2d.strides, vec![1, 4]);

        let shape_3d = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape_3d.shape, vec![2, 3, 4]);
        assert_eq!(shape_3d.strides, vec![12, 4, 1]);

        let shape_3d = shape_3d.permute(&[2, 0, 1]);
        assert_eq!(shape_3d.shape, vec![4, 2, 3]);
        assert_eq!(shape_3d.strides, vec![1, 12, 4]);

        let shape_4d = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape_4d.strides, vec![60, 20, 5, 1]);

        let shape_4d = shape_4d.permute(&[3, 2, 1, 0]);
        assert_eq!(shape_4d.shape, vec![5, 4, 3, 2]);
        assert_eq!(shape_4d.strides, vec![1, 5, 20, 60]);

        let shape_identity = TensorShape::new(vec![2, 3, 4]);
        let original_shape = shape_identity.shape.clone();
        let original_strides = shape_identity.strides.clone();

        let shape_identity = shape_identity.permute(&[0, 1, 2]);
        assert_eq!(shape_identity.shape, original_shape);
        assert_eq!(shape_identity.strides, original_strides);

        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let shape_1d = shape_1d.permute(&[0]);
        assert_eq!(shape_1d.shape, vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let permuted_shape = original_shape.permute(&[1, 2, 0]);

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

        let empty_shape = TensorShape::new(vec![]).permute(&[]);
        assert_eq!(empty_shape.shape, vec![]);
        assert_eq!(empty_shape.strides, vec![]);
    }

    #[test]
    fn test_merge() {
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(shape.strides, vec![60, 20, 5, 1]);

        let merged_shape = shape.merge(1..=2);
        assert_eq!(merged_shape.shape, vec![2, 12, 5]);
        assert_eq!(merged_shape.strides, vec![60, 5, 1]);

        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        let merged_shape = shape.merge(1..=1);
        assert_eq!(merged_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(merged_shape.strides, vec![60, 20, 5, 1]);

        let original_shape = TensorShape::new(vec![2, 3, 4]);
        let merged_shape = original_shape.merge(1..=2);

        assert_eq!(
            original_shape.ravel_index(&[0, 0, 0]),
            merged_shape.ravel_index(&[0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[0, 1, 2]),
            merged_shape.ravel_index(&[0, 6])
        );
        assert_eq!(
            original_shape.ravel_index(&[1, 2, 3]),
            merged_shape.ravel_index(&[1, 11])
        );

        let shape = TensorShape::new(vec![10]);
        let merged_shape = shape.merge(0..=0);
        assert_eq!(merged_shape.shape, vec![10]);
        assert_eq!(merged_shape.strides, vec![1]);
    }

    #[test]
    fn test_split() {
        let shape = TensorShape::new(vec![2, 12, 5]);
        let split_shape = shape.split(1, &[3, 4]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 5]);
        assert_eq!(split_shape.strides, vec![60, 20, 5, 1]);

        let shape = TensorShape::new(vec![24]);
        let split_shape = shape.split(0, &[2, 3, 0]);
        assert_eq!(split_shape.shape, vec![2, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 4, 1]);

        let shape = TensorShape::new(vec![2, 3, 24]);
        let split_shape = shape.split(2, &[4, 6]);
        assert_eq!(split_shape.shape, vec![2, 3, 4, 6]);
        assert_eq!(split_shape.strides, vec![72, 24, 6, 1]);

        let shape = TensorShape::new(vec![12, 3, 4]);
        let split_shape = shape.split(0, &[3, 4]);
        assert_eq!(split_shape.shape, vec![3, 4, 3, 4]);
        assert_eq!(split_shape.strides, vec![48, 12, 4, 1]);

        let shape = TensorShape::new(vec![30]);
        let split_shape = shape.split(0, &[5, 6]);
        assert_eq!(split_shape.shape, vec![5, 6]);
        assert_eq!(split_shape.strides, vec![6, 1]);

        let shape = TensorShape::new(vec![2, 60, 3]);
        let split_shape = shape.split(1, &[4, 0, 5]);
        assert_eq!(split_shape.shape, vec![2, 4, 3, 5, 3]);
        assert_eq!(split_shape.strides, vec![180, 45, 15, 3, 1]);

        let original_shape = TensorShape::new(vec![6, 8]);
        let split_shape = original_shape.split(0, &[2, 3]).split(2, &[4, 2]);

        assert_eq!(
            original_shape.ravel_index(&[0, 0]),
            split_shape.ravel_index(&[0, 0, 0, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[2, 6]),
            split_shape.ravel_index(&[0, 2, 3, 0])
        );
        assert_eq!(
            original_shape.ravel_index(&[5, 7]),
            split_shape.ravel_index(&[1, 2, 3, 1])
        );

        let shape = TensorShape::new(vec![4]);
        let split_shape = shape.split(0, &[4, 1]);
        assert_eq!(split_shape.shape, vec![4, 1]);
        assert_eq!(split_shape.strides, vec![1, 1]);

        let shape = TensorShape::new(vec![2, 3, 4]);
        let split_shape = shape.split(1, &[1, 3]);
        assert_eq!(split_shape.shape, vec![2, 1, 3, 4]);
        assert_eq!(split_shape.strides, vec![12, 12, 4, 1]);
    }

    #[test]
    fn test_slice() {
        let shape = TensorShape::new(vec![5, 6]);
        assert_eq!(shape.shape, vec![5, 6]);
        assert_eq!(shape.strides, vec![6, 1]);
        assert_eq!(shape.linear_offset, 0);

        let sliced_shape = shape.slice(0, 1..=3).slice(1, 2..=4);

        assert_eq!(sliced_shape.shape, vec![3, 3]);
        assert_eq!(sliced_shape.strides, vec![6, 1]);
        assert_eq!(sliced_shape.linear_offset, 1 * 6 + 2 * 1);

        let shape_3d = TensorShape::new(vec![4, 5, 6]);
        assert_eq!(shape_3d.strides, vec![30, 6, 1]);

        let sliced_3d = shape_3d.slice(0, 1..=2).slice(2, 1..=4);

        assert_eq!(sliced_3d.shape, vec![2, 5, 4]);
        assert_eq!(sliced_3d.strides, vec![30, 6, 1]);
        assert_eq!(sliced_3d.linear_offset, 1 * 30 + 1 * 1);

        let shape_1d = TensorShape::new(vec![10]);
        let sliced_1d = shape_1d.slice(0, 3..=7);

        assert_eq!(sliced_1d.shape, vec![5]);
        assert_eq!(sliced_1d.strides, vec![1]);
        assert_eq!(sliced_1d.linear_offset, 3);

        let shape_partial = TensorShape::new(vec![3, 4, 5]);
        let sliced_partial = shape_partial.slice(1, 1..=2);

        assert_eq!(sliced_partial.shape, vec![3, 2, 5]);
        assert_eq!(sliced_partial.strides, vec![20, 5, 1]);
        assert_eq!(sliced_partial.linear_offset, 1 * 5);

        let shape_single = TensorShape::new(vec![5, 5]);
        let sliced_single = shape_single.slice(0, 2..=2).slice(1, 3..=3);

        assert_eq!(sliced_single.shape, vec![1, 1]);
        assert_eq!(sliced_single.strides, vec![5, 1]);
        assert_eq!(sliced_single.linear_offset, 2 * 5 + 3 * 1);

        let original_shape = TensorShape::new(vec![4, 6]);
        let sliced_test = original_shape.slice(0, 1..=2).slice(1, 2..=4);

        let sliced_flat = sliced_test.linear_offset + sliced_test.ravel_index(&[0, 0]);
        let original_flat = original_shape.ravel_index(&[1, 2]);
        assert_eq!(sliced_flat, original_flat);

        let sliced_flat = sliced_test.linear_offset + sliced_test.ravel_index(&[1, 2]);
        let original_flat = original_shape.ravel_index(&[2, 4]);
        assert_eq!(sliced_flat, original_flat);
    }

    #[test]
    fn test_skip() {
        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.strides, vec![1]);

        let skipped_1d = shape_1d.skip(0, 2);
        assert_eq!(skipped_1d.shape, vec![5]);
        assert_eq!(skipped_1d.strides, vec![2]);
        assert_eq!(skipped_1d.linear_offset, 0);

        let shape_1d_odd = TensorShape::new(vec![9]);
        let skipped_1d_odd = shape_1d_odd.skip(0, 2);
        assert_eq!(skipped_1d_odd.shape, vec![5]);
        assert_eq!(skipped_1d_odd.strides, vec![2]);

        let shape_2d = TensorShape::new(vec![6, 8]);
        assert_eq!(shape_2d.strides, vec![8, 1]);

        let skipped_dim0 = shape_2d.skip(0, 2);
        assert_eq!(skipped_dim0.shape, vec![3, 8]);
        assert_eq!(skipped_dim0.strides, vec![16, 1]);
        assert_eq!(skipped_dim0.linear_offset, 0);

        let skipped_dim1 = shape_2d.skip(1, 3);
        assert_eq!(skipped_dim1.shape, vec![6, 3]);
        assert_eq!(skipped_dim1.strides, vec![8, 3]);
        assert_eq!(skipped_dim1.linear_offset, 0);

        let shape_3d = TensorShape::new(vec![4, 6, 8]);
        assert_eq!(shape_3d.strides, vec![48, 8, 1]);

        let skipped_3d = shape_3d.skip(1, 2);
        assert_eq!(skipped_3d.shape, vec![4, 3, 8]);
        assert_eq!(skipped_3d.strides, vec![48, 16, 1]);
        assert_eq!(skipped_3d.linear_offset, 0);

        let shape_chain = TensorShape::new(vec![8, 9]);
        let double_skipped = shape_chain.skip(0, 2).skip(1, 3);
        assert_eq!(double_skipped.shape, vec![4, 3]);
        assert_eq!(double_skipped.strides, vec![18, 3]);
        assert_eq!(double_skipped.linear_offset, 0);

        let shape_noop = TensorShape::new(vec![5, 7]);
        let no_change = shape_noop.skip(0, 1).skip(1, 1);
        assert_eq!(no_change.shape, shape_noop.shape);
        assert_eq!(no_change.strides, shape_noop.strides);
        assert_eq!(no_change.linear_offset, shape_noop.linear_offset);

        let test_cases = vec![(10, 2, 5), (10, 3, 4), (9, 3, 3), (7, 4, 2), (1, 2, 1)];

        for (original_size, step, expected_size) in test_cases {
            let shape = TensorShape::new(vec![original_size]);
            let skipped = shape.skip(0, step);
            assert_eq!(
                skipped.shape[0], expected_size,
                "Failed for {}.div_ceil({}) = {}",
                original_size, step, expected_size
            );
        }

        let shape_with_offset = TensorShape {
            shape: vec![6, 8],
            strides: vec![8, 1],
            linear_offset: 10,
        };

        let skipped_with_offset = shape_with_offset.skip(0, 3);
        assert_eq!(skipped_with_offset.shape, vec![2, 8]);
        assert_eq!(skipped_with_offset.strides, vec![24, 1]);
        assert_eq!(skipped_with_offset.linear_offset, 10);

        let original_shape = TensorShape::new(vec![6, 8]);
        let skipped_shape = original_shape.skip(1, 2);

        assert_eq!(skipped_shape.shape, vec![6, 4]);
        assert_eq!(skipped_shape.strides, vec![8, 2]);

        let skipped_flat = skipped_shape.linear_offset + skipped_shape.ravel_index(&[1, 2]);
        let original_flat = original_shape.ravel_index(&[1, 4]);
        assert_eq!(skipped_flat, original_flat);

        let skipped_flat = skipped_shape.linear_offset + skipped_shape.ravel_index(&[0, 1]);
        let original_flat = original_shape.ravel_index(&[0, 2]);
        assert_eq!(skipped_flat, original_flat);

        let skipped_flat = skipped_shape.linear_offset + skipped_shape.ravel_index(&[2, 3]);
        let original_flat = original_shape.ravel_index(&[2, 6]);
        assert_eq!(skipped_flat, original_flat);
    }

    #[test]
    fn test_map() {
        let storage = TensorStorage {
            data: vec![1, 2, 3, 4, 5],
        };

        let mapped_storage = storage.map(|x| x * 2);
        assert_eq!(mapped_storage.data, vec![2, 4, 6, 8, 10]);

        let float_storage = storage.map(|x| *x as f32 + 0.5);
        assert_eq!(float_storage.data, vec![1.5, 2.5, 3.5, 4.5, 5.5]);

        let mut tensor = Tensor::<i32>::zeros(vec![2, 3]);
        tensor[&[0, 0]] = 1;
        tensor[&[0, 1]] = 2;
        tensor[&[0, 2]] = 3;
        tensor[&[1, 0]] = 4;
        tensor[&[1, 1]] = 5;
        tensor[&[1, 2]] = 6;

        let mapped_tensor = tensor.map(|x| x * x);

        assert_eq!(mapped_tensor.shape.shape, vec![2, 3]);
        assert_eq!(mapped_tensor.shape.strides, vec![3, 1]);

        assert_eq!(mapped_tensor[&[0, 0]], 1); // 1 * 1
        assert_eq!(mapped_tensor[&[0, 1]], 4); // 2 * 2
        assert_eq!(mapped_tensor[&[0, 2]], 9); // 3 * 3
        assert_eq!(mapped_tensor[&[1, 0]], 16); // 4 * 4
        assert_eq!(mapped_tensor[&[1, 1]], 25); // 5 * 5
        assert_eq!(mapped_tensor[&[1, 2]], 36); // 6 * 6

        let string_tensor = tensor.map(|x| format!("value_{}", x));
        assert_eq!(string_tensor[&[0, 0]], "value_1");
        assert_eq!(string_tensor[&[1, 2]], "value_6");

        let bool_tensor = tensor.map(|x| *x > 3);
        assert_eq!(bool_tensor[&[0, 0]], false); // 1 > 3
        assert_eq!(bool_tensor[&[0, 1]], false); // 2 > 3
        assert_eq!(bool_tensor[&[0, 2]], false); // 3 > 3
        assert_eq!(bool_tensor[&[1, 0]], true); // 4 > 3
        assert_eq!(bool_tensor[&[1, 1]], true); // 5 > 3
        assert_eq!(bool_tensor[&[1, 2]], true); // 6 > 3
    }

    #[test]
    fn test_reduce() {
        // Test reducing a 1D tensor (should result in scalar)
        let mut tensor_1d = Tensor::<i32>::zeros(vec![5]);
        tensor_1d[&[0]] = 1;
        tensor_1d[&[1]] = 2;
        tensor_1d[&[2]] = 3;
        tensor_1d[&[3]] = 4;
        tensor_1d[&[4]] = 5;

        let sum_1d = tensor_1d.reduce(0, |a, b| a + b);
        assert_eq!(sum_1d.shape.shape, vec![]); // scalar
        assert_eq!(sum_1d.shape.is_scalar(), true);
        assert_eq!(sum_1d.storage.data, vec![15]); // 1+2+3+4+5 = 15

        let max_1d = tensor_1d.reduce(0, |a, b| if a > b { *a } else { *b });
        assert_eq!(max_1d.shape.shape, vec![]);
        assert_eq!(max_1d.storage.data, vec![5]);

        // Test reducing a 2D tensor along dimension 0 (rows)
        let mut tensor_2d = Tensor::<i32>::zeros(vec![3, 4]);
        // Fill with values: row 0: [1,2,3,4], row 1: [5,6,7,8], row 2: [9,10,11,12]
        for i in 0..3 {
            for j in 0..4 {
                tensor_2d[&[i, j]] = (i * 4 + j + 1) as i32;
            }
        }

        let sum_rows = tensor_2d.reduce(0, |a, b| a + b);
        assert_eq!(sum_rows.shape.shape, vec![4]); // result has shape [4]
        assert_eq!(sum_rows[&[0]], 15); // 1+5+9 = 15
        assert_eq!(sum_rows[&[1]], 18); // 2+6+10 = 18
        assert_eq!(sum_rows[&[2]], 21); // 3+7+11 = 21
        assert_eq!(sum_rows[&[3]], 24); // 4+8+12 = 24

        // Test reducing the same 2D tensor along dimension 1 (columns)
        let sum_cols = tensor_2d.reduce(1, |a, b| a + b);
        assert_eq!(sum_cols.shape.shape, vec![3]); // result has shape [3]
        assert_eq!(sum_cols[&[0]], 10); // 1+2+3+4 = 10
        assert_eq!(sum_cols[&[1]], 26); // 5+6+7+8 = 26
        assert_eq!(sum_cols[&[2]], 42); // 9+10+11+12 = 42

        // Test reducing a 3D tensor
        let mut tensor_3d = Tensor::<f32>::zeros(vec![2, 3, 4]);
        let mut value = 1.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    tensor_3d[&[i, j, k]] = value;
                    value += 1.0;
                }
            }
        }

        // Reduce along dimension 0 (should result in shape [3, 4])
        let reduced_dim0 = tensor_3d.reduce(0, |a, b| a + b);
        assert_eq!(reduced_dim0.shape.shape, vec![3, 4]);
        // First slice [0,:,:] has values 1-12, second slice [1,:,:] has values 13-24
        assert_eq!(reduced_dim0[&[0, 0]], 14.0); // 1.0 + 13.0
        assert_eq!(reduced_dim0[&[0, 1]], 16.0); // 2.0 + 14.0
        assert_eq!(reduced_dim0[&[2, 3]], 36.0); // 12.0 + 24.0

        // Reduce along dimension 1 (should result in shape [2, 4])
        let reduced_dim1 = tensor_3d.reduce(1, |a, b| a + b);
        assert_eq!(reduced_dim1.shape.shape, vec![2, 4]);
        // Sum along the middle dimension
        assert_eq!(reduced_dim1[&[0, 0]], 15.0); // 1.0 + 5.0 + 9.0
        assert_eq!(reduced_dim1[&[0, 3]], 24.0); // 4.0 + 8.0 + 12.0
        assert_eq!(reduced_dim1[&[1, 0]], 51.0); // 13.0 + 17.0 + 21.0

        // Reduce along dimension 2 (should result in shape [2, 3])
        let reduced_dim2 = tensor_3d.reduce(2, |a, b| a + b);
        assert_eq!(reduced_dim2.shape.shape, vec![2, 3]);
        assert_eq!(reduced_dim2[&[0, 0]], 10.0); // 1.0 + 2.0 + 3.0 + 4.0
        assert_eq!(reduced_dim2[&[0, 1]], 26.0); // 5.0 + 6.0 + 7.0 + 8.0
        assert_eq!(reduced_dim2[&[1, 2]], 90.0); // 21.0 + 22.0 + 23.0 + 24.0

        // Test with different reduction functions
        let mut small_tensor = Tensor::<i32>::zeros(vec![2, 3]);
        small_tensor[&[0, 0]] = 5;
        small_tensor[&[0, 1]] = 2;
        small_tensor[&[0, 2]] = 8;
        small_tensor[&[1, 0]] = 1;
        small_tensor[&[1, 1]] = 9;
        small_tensor[&[1, 2]] = 3;

        // Test max reduction
        let max_reduction = small_tensor.reduce(0, |a, b| if a > b { *a } else { *b });
        assert_eq!(max_reduction.shape.shape, vec![3]);
        assert_eq!(max_reduction[&[0]], 5); // max(5, 1)
        assert_eq!(max_reduction[&[1]], 9); // max(2, 9)
        assert_eq!(max_reduction[&[2]], 8); // max(8, 3)

        // Test min reduction
        let min_reduction = small_tensor.reduce(1, |a, b| if a < b { *a } else { *b });
        assert_eq!(min_reduction.shape.shape, vec![2]);
        assert_eq!(min_reduction[&[0]], 2); // min(5, 2, 8)
        assert_eq!(min_reduction[&[1]], 1); // min(1, 9, 3)

        // Test product reduction
        let product_reduction = small_tensor.reduce(0, |a, b| a * b);
        assert_eq!(product_reduction[&[0]], 5); // 5 * 1
        assert_eq!(product_reduction[&[1]], 18); // 2 * 9
        assert_eq!(product_reduction[&[2]], 24); // 8 * 3

        // Test edge case: single element tensor
        let mut single_tensor = Tensor::<i32>::zeros(vec![1]);
        single_tensor.storage.data[0] = 42;
        let single_reduced = single_tensor.reduce(0, |a, b| a + b);
        assert_eq!(single_reduced.shape.shape, vec![]);
        assert_eq!(single_reduced.storage.data, vec![42]);

        // Test edge case: tensor with dimension of size 1
        let mut narrow_tensor = Tensor::<i32>::zeros(vec![1, 5]);
        for j in 0..5 {
            narrow_tensor[&[0, j]] = j as i32 + 1;
        }
        
        let reduced_narrow = narrow_tensor.reduce(0, |a, b| a + b);
        assert_eq!(reduced_narrow.shape.shape, vec![5]);
        for j in 0..5 {
            assert_eq!(reduced_narrow[&[j]], (j as i32) + 1);
        }

        let reduced_narrow2 = narrow_tensor.reduce(1, |a, b| a + b);
        assert_eq!(reduced_narrow2.shape.shape, vec![1]);
        assert_eq!(reduced_narrow2[&[0]], 15); // 1+2+3+4+5
    }
}
