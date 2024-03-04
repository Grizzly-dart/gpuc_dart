import 'package:gpuc_dart/gpuc_dart.dart';

class Size {
  final List<int> _sizes;

  Size(this._sizes);

  int get dims => _sizes.length;

  int operator [](int index) => _sizes[index];
}

abstract class Tensor {
  Size get size;
}

/*
class CPUTensor implements Tensor {}
 */