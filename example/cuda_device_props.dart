import 'package:gpuc_dart/gpuc_dart.dart';

void main() {
  initializeTensorc();
  print(CudaFFI.getDeviceProps(0));
}