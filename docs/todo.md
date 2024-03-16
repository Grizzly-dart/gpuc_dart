+ Tensor.rearrange in C and CUDA (pytorch.permute)
+ Test with complex math equation to optimize intermediate results
+ Split C library into C and CUDA
+ Better print Tensors as tables
+ MaxPool2D return indices
+ tensor.json read and write

# Layers
+ Group normalization
+ ResNet

# Multi device
+ Test seamless switching between devices

# Low prio
+ ROCm implementation
+ SYCL implementation

# Decisions

+ Find a way to keep Tensors in GPU memory between operations
+ Find way to track Tensors in GPU and transfer them to CPU when free memory is not sufficient