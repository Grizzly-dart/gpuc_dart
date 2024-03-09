import 'package:gpuc_dart/gpuc_dart.dart';

void main() async {
  initializeTensorc();
  final t1 = Tensor.random(Dim.twoD(512, 512));
  print(t1[Dim([1])]);
  print('Finished');
}