import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/matrix.dart';

export 'dim.dart';

class Tensor with ListMixin<Tensor> implements Resource {
  String name;

  final NList as1d;

  Dim _size;

  Tensor(this.as1d, this._size, {this.name = '', Context? context}) {
    context?.add(as1d);
    _finalizer.attach(this, as1d);
    if (as1d.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  factory Tensor.fromList(List<double> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    final data = CList.fromList(list, context: context);
    return Tensor(data, size, name: name, context: context);
  }

  factory Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return Tensor(CList.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = CList.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr => as1d.ptr;

  Dim get size => _size;

  int get nel => _size.nel;

  DeviceType get deviceType => as1d.deviceType;

  int get deviceId => as1d.deviceId;

  Device get device => as1d.device;

  double scalar([int index = 0]) => as1d[index];

  void reshape(Dim newSize) {
    if (newSize.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = newSize;
  }

  void squeeze(int dims) {
    _size = _size.squeeze(dims);
  }

  // TODO support squeezing
  List<List<double>> toMatrix() {
    if (_size.dims != 2) {
      throw StateError('Must be a 2D tensor');
    }
    final matrix = <List<double>>[];
    for (var i = 0; i < _size.rows; i++) {
      final row = <double>[];
      for (var j = 0; j < _size.cols; j++) {
        row.add(as1d[i * _size.cols + j]);
      }
      matrix.add(row);
    }
    return matrix;
  }

  set set(Tensor other) {
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    as1d.copyFrom(other.as1d);
  }

  // TODO start and length
  Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    if (index is! Dim) index = Dim.from(index);
    if (_size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(_size.toList().skip(index.dims));
    return Tensor(as1d.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }

  @override
  Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    if (index is! Dim) index = Dim.from(index);
    if (!_size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(_size.asList.skip(index.dims));
    return Tensor(as1d.view(index.nel * outSize.nel, outSize.nel), outSize);
  }

  @override
  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, Tensor value) {
    if (index is! Dim) index = Dim.from(index);
    if (!_size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(_size.asList.skip(index.dims));
    if (value.size.nel != outSize.nel) {
      throw ArgumentError('Size mismatch');
    }

    final view =
        Tensor(as1d.view(index.nel * outSize.nel, outSize.nel), outSize);
    view.set = value;
  }

  NList row(int index, {int colDims = 1}) {
    if (_size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    final size2d = _size.squeeze2D(colDims: colDims);
    if (index < 0 || index >= size2d.rows) {
      throw ArgumentError('Index out of range');
    }
    return as1d.view(index * size2d.cols, size2d.cols);
  }

  Matrix as2d({int colDims = 1}) => Matrix(this, colDims: colDims);

  // TODO auto release inp1 and inp2
  @override
  Tensor operator +(covariant Tensor other) {
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(as1d, stream: stream, context: ctx);
      final inp2 = CudaList.copy(other.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, nel, context: ctx);
      cuda.addition(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${other.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Tensor sumRows({int colDims = 1}) {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = _size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CudaList.copy(as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, outSize.nel, context: ctx);
      cuda.sum2D(stream, out.ptr.cast(), inp.ptr.cast(), inpSize.to2D());
      final outTensor = Tensor.sized(outSize, name: 'sum2D($name)');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  void release() {
    as1d.release();
  }

  bool isEqual(Tensor other, {double epsilon = 1e-8}) {
    int nel = size.nel;
    if (nel > other.size.nel) {
      nel = other.size.nel;
    }
    for (var i = 0; i < nel; i++) {
      if ((as1d[i] - other.as1d[i]).abs() > epsilon) {
        return false;
      }
    }
    return true;
  }

  @override
  String toString() => '$as1d';

  static final _finalizer = Finalizer<NList>((l) {
    l.release();
  });

  @override
  int get length => _size[0];

  @override
  set length(int newLength) {
    if (newLength == 0) {
      throw ArgumentError('Length must be at least 1');
    }
    final newSize = _size.toList();
    newSize[0] = newLength;
    _size = Dim(newSize);
    as1d.length = nel;
  }

  Tensor rearrange(List<int> order, {DeviceType? forceDeviceType}) {
    if (order.length != _size.dims) {
      throw ArgumentError('Invalid order length');
    }
    final outSize = _size.rearrange(order);
    // TODO detect device
    final deviceType = DeviceType.dart;
    if (deviceType == DeviceType.dart) {
      final outData = DartList.sized(outSize.nel);
      for (int i = 0; i < _size.nel; i++) {
        final index = _size.unravel(i);
        final outIndex = index.rearrange(order).ravel;
        outData[outIndex] = as1d[i];
      }
      final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
      return outTensor;
    } else if (deviceType == DeviceType.c) {
      // TODO
    } else if (deviceType == DeviceType.cuda) {
      /* TODO
      final outData = CList.sized(outSize.nel);
      final ctx = Context();
      try {
        int deviceId = 0; // TODO implement device selection
        final stream = CudaStream(deviceId, context: ctx);
        final inp = CudaList.copy(data, stream: stream, context: ctx);
        CudaFFI.rearrange(stream, outData.ptr.cast(), inp.ptr.cast(),
            _size.toList(), outSize.toList());
        final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
        ctx.releaseOnErr(outTensor);
        return outTensor;
      } catch (e) {
        ctx.release(isError: true);
        rethrow;
      } finally {
        ctx.release();
      }
       */
    }
    throw UnimplementedError('Device not implemented');
  }
}
