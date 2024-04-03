import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/src/native/c/ffi.dart';
import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:gpuc_dart/gpuc_dart.dart';

export 'num_type.dart';

void initializeTensorC({String? libPath}) {
  // TODO support processor architecture
  String os;
  if (Platform.isLinux) {
    os = 'linux';
  } else if (Platform.isMacOS) {
    os = 'darwin';
  } else if (Platform.isWindows) {
    os = 'windows';
  } else {
    return;
  }

  String libraryPath;
  if (libPath != null) {
    libraryPath = libPath;
  } else {
    libraryPath = path.join(Directory.current.path, 'lib', 'asset', os);
  }
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  TensorCNativeFFI.initialize(dylib);
}

final TensorCFFI tc = TensorCFFI(null);

class TensorCFFI {
  final TensorCNativeFFI? _native;

  TensorCFFI(this._native);

  TensorCNativeFFI get native => _native ?? TensorCNativeFFI.instance!;

  bool exists() => _native != null || TensorCNativeFFI.instance != null;

  void memcpy(ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) =>
      native.memcpy(dst, src, size);

  ffi.Pointer<T> realloc<T extends ffi.NativeType>(
      ffi.Pointer<T> ptr, int size) {
    final newPtr = native.realloc(ptr.cast(), size);
    if (newPtr == ffi.nullptr) {
      throw CException('Failed to realloc');
    }
    return ptr.cast();
  }

  void binaryArith(List<List<List<COpBinaryArith>>> op, NumPtr out, NumPtr a,
      NumPtr b, int size) {
    final operation = op[out.type.index][a.type.index][b.type.index];
    final err = operation(
        out.ptr.cast(), a.ptr.cast(), b.ptr.cast(), ffi.nullptr, size, 0);
    if (err != ffi.nullptr) {
      throw CException(err.toDartString());
    }
  }

  static final finalizer = Finalizer((other) {
    if (other is ffi.Pointer) {
      ffi.malloc.free(other.cast());
    } else {
      stdout.writeln('Unknown type ${other.runtimeType}');
    }
  });
}

extension TensorCFFITensorExtension on TensorCFFI {
  Tensor binaryArithTensor(
      List<List<List<COpBinaryArith>>> op, Tensor a, Tensor b,
      {Tensor? out}) {
    if (a.nel != b.nel) {
      throw Exception('The number of elements of a and b must be the same');
    }
    if (out != null && out.nel != a.nel) {
      throw Exception(
          'The number of elements of out must be the same as a and b');
    }
    NumType outType =
        out?.type ?? (a.type.bytes > b.type.bytes ? a.type : b.type);
    final size = a.size;
    out ??= Tensor.sized(size, outType, name: '${a.name} + ${b.name}');

    binaryArith(op, out.as1d as COnesor, a.as1d as COnesor, b.as1d as COnesor,
        size.nel);
    return out;
  }
}

class CException implements Exception {
  final String message;

  CException(this.message);

  @override
  String toString() => message;
}
