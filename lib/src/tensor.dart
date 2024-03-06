import 'dart:ffi' as ffi;
import 'dart:math';
import 'package:gpuc_dart/src/clist.dart';
import 'package:gpuc_dart/src/cuda.dart';

abstract class Size {
  factory Size(Iterable<int> sizes) => _SizeImpl(List.from(sizes));

  factory Size.twoD(int rows, [int? cols]) => _SizeImpl([rows, cols ?? rows]);

  int operator [](int index);

  int get dims;

  Size2D get twoD;

  int get nel;

  int get rows;

  int get cols;

  int get channels;

  int get batch;

  List<int> toList();
}

class _SizeImpl implements Size {
  final List<int> _sizes;

  _SizeImpl(this._sizes);

  @override
  int get dims => _sizes.length;

  @override
  int operator [](int index) => _sizes[index];

  @override
  int get nel => _sizes.reduce((a, b) => a * b);

  @override
  int get rows {
    if (dims < 2) {
      return 1;
    }
    return _sizes[dims - 2];
  }

  @override
  int get cols {
    if (dims < 1) {
      throw StateError('Not enough dimensions');
    }
    return _sizes[dims - 1];
  }

  @override
  int get channels {
    if (dims < 3) {
      return 1;
    }
    return _sizes[dims - 3];
  }

  @override
  int get batch {
    if (dims < 4) {
      return 1;
    }
    return _sizes[dims - 4];
  }

  @override
  Size2D get twoD => Size2D(rows: rows, cols: cols);

  @override
  List<int> toList() => List.from(_sizes);
}

class Size2D implements Size {
  @override
  final int rows;
  @override
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

  @override
  List<int> toList() => [rows, cols];

  @override
  int operator [](int index) {
    if (index == 0) {
      return rows;
    } else if (index == 1) {
      return cols;
    }
    throw RangeError('Index out of range');
  }

  @override
  int get dims => 2;

  @override
  int get nel => rows * cols;

  @override
  int get channels => 1;

  @override
  int get batch => 1;

  @override
  Size2D get twoD => this;
}

class Tensor {
  String name;

  NList _data;

  Size _size;

  Tensor(this._data, this._size, {this.name = ''}) {
    _finalizer.attach(this, _data);
    if (_data.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  factory Tensor.empty(Size size,
      {DeviceType deviceType = DeviceType.c,
      int deviceId = 0,
      String name = ''}) {
    return Tensor(
        NList.allocate(size.nel, deviceType: deviceType, deviceId: deviceId),
        size,
        name: name);
  }

  factory Tensor.random(Size size, {Random? random, String name = ''}) {
    random ??= Random();
    final data = CList.allocate(size.nel);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name);
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
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    if (other.deviceType != deviceType) {
      // TODO handle memory transfer
      throw UnimplementedError('Device mismatch');
    }
    final out = Tensor.empty(size,
        deviceType: DeviceType.cuda,
        deviceId: deviceId,
        name: '$name + ${other.name}');
    final inp1 = to(DeviceType.cuda);
    final inp2 = other.to(DeviceType.cuda);
    CudaFFIFunctions.addition(
        out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
    return out;
  }

  Tensor to(DeviceType device, {int deviceId = 0}) {
    if (_data.deviceType == device && _data.deviceId == deviceId) {
      return this;
    }
    return Tensor(NList.copy(_data, device: device, deviceId: deviceId), _size);
  }

  List toList() {
    final list = <double>[];
    for (var i = 0; i < nel; i++) {
      list.add(_data[i]);
    }
    return list;
  }

  static final _finalizer = Finalizer<NList>((l) {
    print('releasing tensor');
    l.release();
  });
}
