import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/src/core/c.dart';
import 'package:gpuc_dart/src/core/dart_list.dart';
import 'package:gpuc_dart/src/core/releaseable.dart';
import 'package:gpuc_dart/src/native/cuda.dart';
import 'package:path/path.dart' as path;

void initializeTensorc() {
  String libraryPath = path.join(Directory.current.path, 'lib', 'asset');
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libtensorc.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorc.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CListFFIFunctions.initialize(dylib);
  CudaFFI.initialize(dylib);
}

abstract class NList implements Resource {
  DeviceType get deviceType;

  int get deviceId;

  // TODO this can be late final
  Device get device => Device(deviceType, deviceId);

  int get length;

  int get lengthBytes;

  double operator [](int index);

  void operator []=(int index, double value);

  ffi.Pointer<ffi.Double> get ptr;

  @override
  void release();

  // TODO subview

  // TODO implement partial write
  void copyFrom(NList src);

  // TODO implement partial read
  void copyTo(NList dst);

  CList read({Context? context});

  // TODO sum

  List<double> toList() {
    final list = List<double>.filled(length, 0);
    copyTo(DartList.fromList(list));
    return list;
  }

/*
  static NList allocate(int length,
      {DeviceType deviceType = DeviceType.c,
      int deviceId = 0,
      Context? context}) {
    switch (deviceType) {
      case DeviceType.c:
        return CList.allocate(length, context: context);
      case DeviceType.dart:
        return DartList.fromList(List.filled(length, 0));
      case DeviceType.cuda:
        return CudaList.allocate(length, deviceId: 0, context: context);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }
   */

/*
  static NList copy(NList other,
      {DeviceType device = DeviceType.c, int deviceId = 0, Context? context}) {
    switch (device) {
      case DeviceType.c:
        return CList.copy(other, context: context);
      case DeviceType.dart:
        return DartList.copy(other);
      case DeviceType.cuda:
        return CudaList.copy(other, deviceId: deviceId, context: context);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }
   */
}

class CudaList extends NList {
  ffi.Pointer<ffi.Double> _mem;
  @override
  final int length;
  final int _deviceId;

  CudaList._(this._mem, this.length, this._deviceId, {Context? context}) {
    context?.add(this);
  }

  static CudaList allocate(CudaStream stream, int length, {Context? context}) {
    final lContext = Context();
    final ptr = ffi.calloc
        .allocate<ffi.Pointer<ffi.Void>>(ffi.sizeOf<ffi.Pointer<ffi.Void>>());
    try {
      final err = CudaFFI.allocate(stream.ptr, ptr, length * byteSize);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
      return CudaList._(ptr.value.cast(), length, stream.deviceId,
          context: context);
    } finally {
      lContext.release();
      ffi.calloc.free(ptr); // TODO use context to free
    }
  }

  static CudaList fromList(CudaStream stream, List<double> list,
      {Context? context}) {
    final ret = CudaList.allocate(stream, list.length, context: context);
    ret.copyFrom(DartList.fromList(list), stream: stream);
    return ret;
  }

  static CudaList copy(NList other, {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = CudaList.allocate(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  DeviceType get deviceType => DeviceType.cuda;

  @override
  int get deviceId => _deviceId;

  @override
  int get lengthBytes => length * 8;

  @override
  double operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return CudaFFI.getDouble(_mem, index, deviceId);
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    CudaFFI.setDouble(_mem, index, value, deviceId);
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _mem;

  @override
  void release() {
    if (_mem == ffi.nullptr) return;
    final stream = CudaStream(deviceId);
    try {
      CudaFFI.release(stream.ptr, _mem.cast());
      _mem = ffi.nullptr;
    } finally {
      stream.release();
    }
  }

  @override
  void copyFrom(NList src, {CudaStream? stream}) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: context);
      src = src is CList ? src : src.read(context: context);
      CudaFFI.memcpy(stream, _mem.cast(), src.ptr.cast(), lengthBytes);
    } finally {
      context.release();
    }
  }

  @override
  void copyTo(NList dst, {CudaStream? stream}) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    stream = stream ?? CudaStream(deviceId, context: context);
    try {
      if (dst is CList) {
        CudaFFI.memcpy(
            stream, dst.ptr.cast(), _mem.cast(), dst.lengthBytes);
        return;
      }
      final cSrc = read(context: context, stream: stream);
      dst.copyFrom(cSrc);
    } finally {
      context.release();
    }
  }

  @override
  CList read({Context? context, CudaStream? stream}) {
    final clist = CList.allocate(length, context: context);
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: lContext);
      CudaFFI.memcpy(
          stream, clist.ptr.cast(), _mem.cast(), clist.lengthBytes);
      return clist;
    } finally {
      lContext.release();
    }
  }

  static const byteSize = 8;
}

enum DeviceType { c, dart, cuda, rocm, sycl }

class Device {
  final DeviceType type;
  final int id;

  Device(this.type, this.id);

  @override
  bool operator ==(Object other) {
    if (other is! Device) return false;
    if (identical(this, other)) return true;
    if (type != other.type) return false;
    if (type == DeviceType.c || type == DeviceType.dart) return true;
    return type == other.type && id == other.id;
  }

  @override
  int get hashCode => Object.hashAll([type.index, id]);
}

enum PadMode { constant, reflect, replicate, circular }
