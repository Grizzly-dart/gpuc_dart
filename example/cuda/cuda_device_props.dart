import 'package:gpuc_dart/gpuc_dart.dart';

void main() {
  initializeNativeTensorLibrary();
  print(cuda.getDeviceProps(0));
}