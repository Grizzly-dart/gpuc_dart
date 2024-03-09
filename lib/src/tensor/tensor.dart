import 'dart:ffi' as ffi;
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';

export 'dim.dart';

class Tensor implements Resource {
  String name;

  final CList data;

  Dim _size;

  Tensor(this.data, this._size, {this.name = '', Context? context}) {
    context?.add(data);
    _finalizer.attach(this, data);
    if (data.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  factory Tensor.sized(Dim size, {String name = '', Context? context}) {
    return Tensor(CList.allocate(size.nel, context: context), size,
        name: name, context: context);
  }

  factory Tensor.random(Dim size,
      {Random? random, String name = '', Context? context}) {
    random ??= Random();
    final data = CList.allocate(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr => data.ptr;

  Dim get size => _size;

  int get nel => _size.nel;

  DeviceType get deviceType => data.deviceType;

  int get deviceId => data.deviceId;

  Device get device => data.device;

  double scalar() => data[0];

  void reshape(Dim newSize) {
    if (newSize.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = newSize;
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
        row.add(data[i * _size.cols + j]);
      }
      matrix.add(row);
    }
    return matrix;
  }

  // TODO start and length
  Tensor slice(/* Dim | int | Iterable<int> */ i, {Context? context}) {
    final index = Dim.from(i);
    if (index.dims > _size.dims) {
      throw ArgumentError('Index dimension must be less than tensor dimension');
    }

    final outSize = Dim(_size.toList().sublist(index.dims));

    return Tensor(data.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }

  Tensor operator [](/* Dim | int | Iterable<int> */ i) {
    final index = Dim.from(i);
    if (index.dims > _size.dims) {
      throw ArgumentError('Index dimension must be less than tensor dimension');
    }

    final outSize = Dim(_size.toList().sublist(index.dims));

    return Tensor(data.view(index.nel * outSize.nel, outSize.nel), outSize);
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
      CudaFFI.addition(
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
    Dim outSize = Dim(size.toList().take(size.dims - 1));
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CudaList.copy(data, stream: stream, context: ctx);
      final out = CudaList.allocate(stream, outSize.nel, context: ctx);
      CudaFFI.sum2D(stream, out.ptr.cast(), inp.ptr.cast(),
          Dim2(rows: outSize.nel, cols: size.cols));
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

  bool isEqual(Tensor other, {double epsilon = 1e-8}) {
    int nel = size.nel;
    if (nel > other.size.nel) {
      nel = other.size.nel;
    }
    for (var i = 0; i < nel; i++) {
      if ((data[i] - other.data[i]).abs() > epsilon) {
        return false;
      }
    }
    return true;
  }

  @override
  String toString() => '<$name, $size>${toList()}';

  static final _finalizer = Finalizer<NList>((l) {
    l.release();
  });
}
