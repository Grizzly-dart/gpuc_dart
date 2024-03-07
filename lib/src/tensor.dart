import 'dart:ffi' as ffi;
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';

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

class Tensor implements Resource {
  String name;

  final CList data;

  Size _size;

  Tensor(this.data, this._size, {this.name = '', Context? context}) {
    context?.add(data);
    _finalizer.attach(this, data);
    if (data.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  factory Tensor.sized(Size size, {String name = '', Context? context}) {
    return Tensor(CList.allocate(size.nel, context: context), size,
        name: name, context: context);
  }

  factory Tensor.random(Size size,
      {Random? random, String name = '', Context? context}) {
    random ??= Random();
    final data = CList.allocate(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr => data.ptr;

  Size get size => _size;

  int get nel => _size.nel;

  DeviceType get deviceType => data.deviceType;

  int get deviceId => data.deviceId;

  Device get device => data.device;

  void reshape(Size newSize) {
    if (newSize.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = newSize;
  }

  // TODO auto release inp1 and inp2
  Tensor operator +(Tensor other) {
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(data, stream: stream, context: ctx);
      final inp2 = CudaList.copy(other.data, stream: stream, context: ctx);
      final out = CudaList.allocate(stream, nel, context: ctx);
      CudaFFIFunctions.addition(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${other.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.data, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Tensor sumRows() {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Size outSize = Size(size.toList().take(size.dims - 1));
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CudaList.copy(data, stream: stream, context: ctx);
      final out = CudaList.allocate(stream, outSize.nel, context: ctx);
      CudaFFIFunctions.sum2D(stream, out.ptr.cast(), inp.ptr.cast(),
          Size2D(rows: outSize.nel, cols: size.cols));
      final outTensor = Tensor.sized(outSize, name: 'sum2D($name)');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.data, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  List<double> toList() {
    final list = <double>[];
    for (var i = 0; i < nel; i++) {
      list.add(data[i]);
    }
    return list;
  }

  @override
  void release() {
    data.release();
  }

  static final _finalizer = Finalizer<NList>((l) {
    l.release();
  });
}
