import 'package:gpuc_dart/gpuc_dart.dart';

void printMemInfo(int device) {
  final memInfo = CudaFFIFunctions.getMemInfo(device);
  print('MemInfo:${memInfo.toHumanString()}');
}

void test() {
  final watch = Stopwatch()..start();
  print(CudaFFIFunctions.getDeviceProps(0));
  printMemInfo(0);
  final t1 = Tensor.random(Size.twoD(512, 512));
  final t3 = t1.sumRows();
  printMemInfo(0);
  print(t3.toList());
  printMemInfo(0);
  print(watch.elapsed);
}

void main() async {
  initializeTensorc();
  for(int i = 0; i < 10; i++) {
    test();
    printMemInfo(0);
    await Future.delayed(Duration(seconds: 2));
  }
  print('Finished');
}