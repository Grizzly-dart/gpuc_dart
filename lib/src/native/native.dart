export 'c.dart';
export 'cuda.dart';
import 'package:gpuc_dart/gpuc_dart.dart';

void initializeNativeTensorLibrary(
    {String? tensorCPath, String? tensorCudaPath}) {
  initializeTensorC(libPath: tensorCPath);
  initializeTensorCuda(libPath: tensorCudaPath);
}
