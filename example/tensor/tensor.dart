import 'package:gpuc_dart/gpuc_dart.dart';

void main() async {
  initializeNativeTensorLibrary();
  final t1 = F64Tensor.fromList(List.generate(16, (i) => i.toDouble()),
      size: Dim([4, 4]));
  print(t1[0]);
  print(t1[1]);
  print(t1[2]);
  print(t1[3]);

  print(t1[0].as1d);
  print(t1[0].size);

  await index();

  print('Finished');
}

Future<void> index() async {
  final t1 =
      U16Tensor.generate(Dim([2, 4, 4]), (size, index) => size.ravel(index));
  t1.printTextTable();
  print('-----');
  t1[Extent(Dim([1,2]), Dim([1,3]))].printTextTable();
}
