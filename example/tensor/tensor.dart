import 'package:gpuc_dart/gpuc_dart.dart';

void main() async {
  initializeNativeTensorLibrary();
  final t1 = F64Tensor.fromList(List.generate(16, (i) => i.toDouble()),
      size: Dim([4, 4]));
  print(t1[0]);
  print(t1[1]);
  print(t1[2]);
  print(t1[3]);
  print('Finished');

  print(t1[0].as1d);
  print(t1[0].size);
}
