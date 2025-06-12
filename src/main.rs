use num_traits::Zero;

#[derive(Debug, Clone, PartialEq)]
struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    fn size(&self) -> usize {
        self.shape.iter().product()
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
}
