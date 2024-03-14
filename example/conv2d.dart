import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn2d/maxpool2d.dart';

void main() {
  initializeNativeTensorLibrary();
  final t1 = Tensor.fromList(
      List.generate(3 * 3, (index) => index.toDouble() + 1),
      size: Dim2(3, 3));
  final kernel = Tensor.fromList(
      List.generate(3 * 3, (index) => index.toDouble() + 1),
      size: Dim([1, 1, 3, 3]));
  final conv2D = Conv2D.own(kernel);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
}
