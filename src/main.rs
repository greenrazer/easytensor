use num_traits::Zero;

#[derive(Debug, Clone, PartialEq)]
struct TensorShape {
    shape: Vec<usize>,
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

impl TensorShape {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<T: Zero + Clone> Tensor<T> {
    fn zeros(shape: Vec<usize>) -> Self {
        let ts = TensorShape { shape };
        let size = ts.size();
        Tensor {
            shape: ts,
            storage: TensorStorage {
                data: vec![T::zero(); size],
            },
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