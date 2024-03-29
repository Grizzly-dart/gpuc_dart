+ Finalize CudaStream
+ Onesors should store CPtr and should not use context? Is there a way to mix CPtr and context?
+ Null CudaStream
+ Implement List methods on CuOnesor to not access single elements

+ pow, log, abs, asinh, acosh, atanh, erf, ceil, floor, round
+ mean, std, var, sum, prod, min, max, argmin, argmax
+ safetensors

+ Tensor.rearrange in C and CUDA (pytorch.permute)
+ Test with complex math equation to optimize intermediate results
+ MaxPool2D return indices
+ tensor.json read and write
+ Strided Tensor views
+ Activation functions
+ ResNet
+ Fix printing tensor tables
+ Layer backwards
+ COnesor should not use context but use native finalizer
+ Implement null CudaStream all the way to native

# Layers
+ Group normalization

# Multi device
+ Test seamless switching between devices

# Low prio
+ ROCm implementation
+ SYCL implementation

# Decisions

+ Find a way to keep Tensors in GPU memory between operations
+ Find way to track Tensors in GPU and transfer them to CPU when free memory is not sufficient