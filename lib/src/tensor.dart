import 'dart:ffi' as ffi;
import 'package:gpuc_dart/src/clist.dart';
import 'package:gpuc_dart/src/cuda.dart';

class Size {
  final List<int> _sizes;

  Size(this._sizes);

  int get dims => _sizes.length;

  int operator [](int index) => _sizes[index];

  int get nel => _sizes.reduce((a, b) => a * b);

  int get rows {
    if (dims < 2) {
      return 1;
    }
    return _sizes[dims - 2];
  }

  int get cols {
    if (dims < 1) {
      throw StateError('Not enough dimensions');
    }
    return _sizes[dims - 1];
  }

  int get channels {
    if (dims < 3) {
      return 1;
    }
    return _sizes[dims - 3];
  }

  int get batch {
    if (dims < 4) {
      return 1;
    }
    return _sizes[dims - 4];
  }

  Size2D get twoD => Size2D(rows: rows, cols: cols);

  List<int> toList() => List.from(_sizes);
}

class Size2D {
  final int rows;
  final int cols;

  const Size2D({required this.rows, required this.cols});

  Size2D operator +(/* Size2D | int */ other) {
    if (other is int) {
      return Size2D(rows: rows + other, cols: cols + other);
    } else if (other is Size2D) {
      return Size2D(rows: rows + other.rows, cols: cols + other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Size2D operator -(/* Size2D | int */ other) {
    if (other is int) {
      return Size2D(rows: rows - other, cols: cols - other);
    } else if (other is Size2D) {
      return Size2D(rows: rows - other.rows, cols: cols - other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Size2D operator *(/* int | Size2D */ other) {
    if (other is int) {
      return Size2D(rows: rows * other, cols: cols * other);
    } else if (other is Size2D) {
      return Size2D(rows: rows * other.rows, cols: cols * other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Size2D operator ~/(/* Size2D | int */ scalar) {
    if (scalar is int) {
      return Size2D(rows: rows ~/ scalar, cols: cols ~/ scalar);
    } else if (scalar is Size2D) {
      return Size2D(rows: rows ~/ scalar.rows, cols: cols ~/ scalar.cols);
    }
    throw ArgumentError('Invalid type');
  }

  List<int> toList() => [rows, cols];
}

class Tensor {
  NList _data;

  Size _size;

  Tensor(this._data, this._size) {
    if (_data.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  factory Tensor.empty(Size size,
      {DeviceType deviceType = DeviceType.c, int deviceId = 0}) {
    return Tensor(
        NList.allocate(size.nel, deviceType: deviceType, deviceId: deviceId),
        size);
  }

  ffi.Pointer<ffi.Double> get ptr => _data.ptr;

  Size get size => _size;

  int get nel => _size.nel;

  DeviceType get deviceType => _data.deviceType;

  int get deviceId => _data.deviceId;

  Device get device => _data.device;

  void reshape(Size newSize) {
    if (newSize.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = newSize;
  }

  Tensor operator +(Tensor other) {
    if (other.deviceType != deviceType) {
      // TODO handle memory transfer
      throw UnimplementedError('Device mismatch');
    }
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final out = Tensor.empty(size, deviceType: deviceType, deviceId: deviceId);
    CudaFFIFunctions.addition(
        out.ptr.cast(), ptr.cast(), other.ptr.cast(), nel);
    return out;
  }

  Tensor to(DeviceType device, {int deviceId = 0}) {
    if (_data.deviceType == device && _data.deviceId == deviceId) {
      return this;
    }
    return Tensor(NList.copy(_data, device: device, deviceId: deviceId), _size);
  }
}

abstract class Layer2D {
  Tensor forward(Tensor input);
}

class MaxPool2D implements Layer2D {
  final Size2D kernelSize;

  final Size2D stride;

  final Size2D padding;

  final double padValue;

  final PadMode padMode;

  final Size2D dilation;

  // TODO return indices

  MaxPool2D(this.kernelSize,
      {this.stride = const Size2D(rows: 1, cols: 1),
      this.padding = const Size2D(rows: 0, cols: 0),
      this.padValue = 0,
      this.padMode = PadMode.constant,
      this.dilation = const Size2D(rows: 1, cols: 1)}) {
    // TODO validate
  }

  @override
  Tensor forward(Tensor inp) {
    // TODO validate
    // TODO if multiple devices are available try to parallelize across devices
    if (inp.deviceType == DeviceType.cuda) {
      final out = Tensor.empty(outSize(inp.size),
          deviceType: inp.deviceType, deviceId: inp.deviceId);
      CudaFFIFunctions.maxpool2D(out, inp, kernelSize,
          stride: stride,
          dilation: dilation,
          padding: padding,
          padMode: padMode,
          padValue: padValue);
      return out;
    }
    throw UnimplementedError();
  }

  Size2D outSize2D(Size inSize) {
    // TODO is this the right calculation?
    return inSize.twoD -
        (dilation * (kernelSize - 1)) +
        (padding * 2) ~/ stride +
        Size2D(rows: 1, cols: 1);
  }

  Size outSize(Size inSize) {
    return Size([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
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
