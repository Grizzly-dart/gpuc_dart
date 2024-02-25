import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

import 'package:path/path.dart' as path;

void init() {
  String libraryPath = path.join(Directory.current.path, 'lib', 'asset');
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libgpuc_cuda.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libhello.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'hello.dll');
  } else {
    throw Exception('Unsupported platform');
  }
  final dylib = ffi.DynamicLibrary.open(libraryPath);
  _CudaTensorMethods.make1D = dylib.lookupFunction<
      _CudaTensor Function(ffi.Uint64),
      _CudaTensor Function(int)>('makeTensor1D');
  _CudaTensorMethods.release = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor),
      void Function(_CudaTensor)>('makeTensor1D');

  _CudaTensorMethods.writeTensor = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor, ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(_CudaTensor, ffi.Pointer<ffi.Double>, int)>('writeTensor');
  _CudaTensorMethods.readTensor = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor, ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(_CudaTensor, ffi.Pointer<ffi.Double>, int)>('readTensor');

  elementwiseAdd2 = dylib.lookupFunction<
      ffi.Void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, int)>('elementwiseAdd2');
}

final class _CudaTensor extends ffi.Struct {
  external final ffi.Pointer<ffi.Double> mem;
  @ffi.Uint64()
  external int x;
}

abstract class _CudaTensorMethods {
  static late final _CudaTensor Function(int size) make1D;
  static late final void Function(_CudaTensor tensor) release;

  static late final void Function(
      _CudaTensor tensor, ffi.Pointer<ffi.Double> data, int size) writeTensor;
  static late final void Function(
      _CudaTensor tensor, ffi.Pointer<ffi.Double> data, int size) readTensor;
}

class CBuffer<T extends ffi.NativeType> {
  final ffi.Pointer<T> _ptr;
  final int _length;
  final ffi.Allocator _allocator;

  int get length => _length;

  bool _freed = false;

  CBuffer._(this._ptr, this._length, this._allocator);

  static CBuffer<ffi.Double> allocateDouble(int length,
      {ffi.Allocator allocator = ffi.malloc}) {
    final ptr =
        allocator.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>() * length);
    return CBuffer._(ptr, length, allocator);
  }

  void release() {
    if (_freed) return;

    _allocator.free(_ptr);
    _freed = true;
  }
}

class CudaTensor implements Tensor {
  final _CudaTensor _inner;

  CudaTensor._(this._inner);

  ffi.Pointer<ffi.Double> get arrayPtr => _inner.mem;

  int get length => _inner.x;

  void write(List<double> data) {
    ffi.using((p0) {
      if (data.length != _inner.x) throw Exception('Invalid size; ${_inner.x}');

      final ptr = ffi.malloc
          .allocate<ffi.Double>(ffi.sizeOf<ffi.Double>() * data.length);
      ptr.asTypedList(data.length).setAll(0, data);
      _CudaTensorMethods.writeTensor(_inner, ptr, data.length);
    }, ffi.malloc);
  }

  List<double> toList() {
    final ret = List<double>.filled(_inner.x, 0);
    ffi.using((p0) {
      final ptr =
          ffi.malloc.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>() * _inner.x);
      _CudaTensorMethods.readTensor(_inner, ptr, _inner.x);
      ret.setAll(0, ptr.asTypedList(_inner.x));
    }, ffi.malloc);
    return ret;
  }

  void release() {
    _CudaTensorMethods.release(_inner);
  }

  factory CudaTensor.make1D(int size) =>
      CudaTensor._(_CudaTensorMethods.make1D(size));
}

late final void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>, int) elementwiseAdd2;

abstract class Tensor {}
