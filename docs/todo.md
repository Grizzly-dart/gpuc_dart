+ Link Resource to eachother cascading release (Resource is Context?)
+ Implement List methods on CuOnesor to not access single elements
+ Use mangled C++ functions instead of C functions
+ Finalize/release CuOnesor

+ pow, log, abs, asinh, acosh, atanh, erf, ceil, floor, round
+ mean, std, var, sum, prod, min, max, argmin, argmax
+ safetensors

+ Tensor.rearrange in C and CUDA (pytorch.permute)
+ MaxPool2D return indices
+ tensor.json read and write
+ Strided Tensor views
+ Activation functions
+ ResNet
+ Fix printing tensor tables

# Layers
+ Group normalization

# Multi device
+ Test seamless switching between devices

# Low prio
+ ROCm implementation
+ SYCL implementation

# Decisions

+ Test with complex math equation to optimize intermediate results
+ Find a way to keep Tensors in GPU memory between operations
+ Use zones for device selection?