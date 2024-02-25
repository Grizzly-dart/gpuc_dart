import 'package:gpuc_dart/gpuc_dart.dart';

void main() async {
  init();
  final t1 = CudaTensor.make1D(5)..write([1, 2, 3, 4, 5]);
  final t2 = CudaTensor.make1D(5)..write([1, 2, 3, 4, 5]);
  final t3 = CudaTensor.make1D(5);
  elementwiseAdd2(t3.arrayPtr, t1.arrayPtr, t2.arrayPtr, 5);

  print(t3.toList());

  t1.release();
  t2.release();
  t3.release();
}
