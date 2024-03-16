import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  final a = Tensor.generate(Dim2(2, 2), (i) => i.ravel + 1);
  final b = Tensor.generate(Dim2(2, 2), (i) => i.ravel + 1);
  final out = await a.matmul(b);
  out.printTextTable();
}