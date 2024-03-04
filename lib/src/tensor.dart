import 'package:gpuc_dart/src/clist.dart';

class Size {
  final List<int> _sizes;

  Size(this._sizes);

  int get dims => _sizes.length;

  int operator [](int index) => _sizes[index];

  int get nel => _sizes.reduce((a, b) => a * b);
}

class Tensor {
  CList _data;

  Size _size;

  Tensor(this._data, this._size) {
    if (_data.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  Size get size => _size;

  void reshape(Size newSize) {
    if (newSize.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = newSize;
  }


}

class Conv2D {
  final Tensor _weight;

  final Tensor _bias;

  Conv2D._(this._weight, this._bias);

  factory Conv2D(int inChannels, int outChannels, int kernelSize) {
    final weight =
        CList.allocate(inChannels * outChannels * kernelSize * kernelSize);
    final bias = CList.allocate(outChannels);
    return Conv2D._(
        Tensor(weight, Size([outChannels, inChannels, kernelSize, kernelSize])),
        Tensor(bias, Size([outChannels])));
  }

// TODO add
}
