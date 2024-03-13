import 'package:gpuc_dart/gpuc_dart.dart';

void main() {
  initializeNativeTensorLibrary();
  print(CudaFFI.getDeviceProps(0));
}