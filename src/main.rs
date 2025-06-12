use num_traits::Zero;

#[derive(Debug, Clone, PartialEq)]
struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn ravel_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Indices length does not match tensor shape dimensions.");
        }
        
        indices.iter().zip(self.shape.iter())
            .rev()
            .scan(1, |stride, (&idx, &dim_size)| {
                let result = idx * *stride;
                *stride *= dim_size;
                Some(result)
            })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TensorStorage<T> {
    data: Vec<T>,
}

#[derive(Debug, Clone, PartialEq)]
struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T: Zero + Clone> Tensor<T> {
    fn zeros(shape: Vec<usize>) -> Self {
        let shape = TensorShape { shape };
        let storage = TensorStorage::<T>::zeros(&shape);
        Tensor { shape, storage }
    }
}

impl<T: Zero + Clone> TensorStorage<T> {
    fn zeros(shape: &TensorShape) -> Self {
        TensorStorage {
            data: vec![T::zero(); shape.size()],
        }
    }
}

fn main() {
    let tensor = Tensor::<f32>::zeros(vec![2, 3]);
    println!("{:?}", tensor);
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
        let shape = TensorShape { shape: vec![5] };
        
        assert_eq!(shape.ravel_index(&[0]), 0);
        assert_eq!(shape.ravel_index(&[1]), 1);
        assert_eq!(shape.ravel_index(&[2]), 2);
        assert_eq!(shape.ravel_index(&[3]), 3);
        assert_eq!(shape.ravel_index(&[4]), 4);

        let shape = TensorShape { shape: vec![2, 3] };
        
        assert_eq!(shape.ravel_index(&[0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 2]), 2);
        assert_eq!(shape.ravel_index(&[1, 0]), 3);
        assert_eq!(shape.ravel_index(&[1, 1]), 4);
        assert_eq!(shape.ravel_index(&[1, 2]), 5);

        let shape = TensorShape { shape: vec![2, 3, 4] };
        
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

        let shape = TensorShape { shape: vec![2, 2, 2, 2] };
    
        assert_eq!(shape.ravel_index(&[0, 0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 0]), 2);
        assert_eq!(shape.ravel_index(&[0, 0, 1, 1]), 3);
        assert_eq!(shape.ravel_index(&[0, 1, 0, 0]), 4);
        assert_eq!(shape.ravel_index(&[1, 0, 0, 0]), 8);
        assert_eq!(shape.ravel_index(&[1, 1, 1, 1]), 15);

        let shape = TensorShape { shape: vec![10, 20, 30] };
        
        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(shape.ravel_index(&[0, 0, 1]), 1);
        assert_eq!(shape.ravel_index(&[0, 1, 0]), 30);
        assert_eq!(shape.ravel_index(&[1, 0, 0]), 600);
        assert_eq!(shape.ravel_index(&[5, 10, 15]), 5 * 600 + 10 * 30 + 15);
        assert_eq!(shape.ravel_index(&[9, 19, 29]), 9 * 600 + 19 * 30 + 29);

        let shape = TensorShape { shape: vec![1, 1, 1] };
        assert_eq!(shape.ravel_index(&[0, 0, 0]), 0);

        let shape = TensorShape { shape: vec![3, 4] };
        
        let expected = [
            0, 1, 2, 3, 
            4, 5, 6, 7, 
            8, 9, 10, 11
        ];
        
        let mut index = 0;
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(shape.ravel_index(&[i, j]), expected[index]);
                index += 1;
            }
        }
    }
}
