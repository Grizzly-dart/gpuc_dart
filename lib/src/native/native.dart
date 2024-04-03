export 'c/c.dart';
export 'cuda/cuda.dart';
export 'tenson/tenson.dart';

import 'package:gpuc_dart/gpuc_dart.dart';

void initializeNativeTensorLibrary(
    {String? tensorCPath, String? tensorCudaPath}) {
  initializeTensorC(libPath: tensorCPath);
  initializeTensorCuda(libPath: tensorCudaPath);
}

enum PadMode { constant, reflect, replicate, circular }
