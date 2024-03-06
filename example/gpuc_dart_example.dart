import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/cuda.dart';

void test() {
  final watch = Stopwatch()..start();
  print(CudaFFIFunctions.getDeviceProps(0));
  final t1 = Tensor.random(Size.twoD(512, 512));
  final t2 = Tensor.random(Size.twoD(512, 512));
  final t3 = t1 + t2;
  t3.to(DeviceType.c);
  print(watch.elapsed);
  // print(t3.toList());
}

void main() async {
  initializeTensorc();
  test();
  await Future.delayed(Duration(seconds: 100));
}
