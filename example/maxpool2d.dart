import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn/maxpool2d.dart';

void main() {
  initializeTensorc();
  final t1 = Tensor.fromList(
      List.generate(16 * 16, (index) => index.toDouble()),
      size: Dim2(16, 16));
  // print(t1.as1d);
  final maxPool2D = MaxPool2D(Dim2(3, 3));
  final t2 = maxPool2D.forward(t1);
  print(t2.as1d);
}
