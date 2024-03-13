export 'c.dart';
export 'cuda/cuda.dart';
import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

void initializeNativeTensorLibrary(
    {String? tensorCPath, String? tensorCudaPath}) {
  initializeTensorC(libPath: tensorCPath);
  initializeTensorCuda(libPath: tensorCudaPath);
}
