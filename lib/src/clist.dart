import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'dart:typed_data';
import 'package:path/path.dart' as path;

import 'package:gpuc_dart/src/tensor.dart';

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
  CudaFFIFunctions.initialize(dylib);

  /*
  elementwiseAdd2 = dylib.lookupFunction<
      ffi.Void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, int)>('elementwiseAdd2');
   */
}

abstract class NList {
  DeviceType get deviceType;

  int get deviceId;

  // TODO this can be late final
  Device get device => Device(deviceType, deviceId);

  int get length;

  int get lengthBytes;

  double operator [](int index);

  void operator []=(int index, double value);

  ffi.Pointer<ffi.Double> get ptr;

  void release();

  // TODO subview

  // TODO implement partial write
  void copyFrom(NList src);

  // TODO implement partial read
  void copyTo(NList dst);

  CList read();

  static NList allocate(int length,
      {DeviceType deviceType = DeviceType.c, int deviceId = 0}) {
    switch (deviceType) {
      case DeviceType.c:
        return CList.allocate(length);
      case DeviceType.dart:
        return DartList.fromList(List.filled(length, 0));
      case DeviceType.cuda:
        return CudaList.allocate(length, deviceId: 0);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }

  static NList copy(NList other,
      {DeviceType device = DeviceType.c, int deviceId = 0}) {
    switch (device) {
      case DeviceType.c:
        return CList.copy(other);
      case DeviceType.dart:
        return DartList.copy(other);
      case DeviceType.cuda:
        return CudaList.copy(other, deviceId: deviceId);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }
}

class DartList extends NList {
  final List<double> _list;

  DartList.fromList(this._list);

  static DartList copy(NList other) {
    if (other is DartList) {
      final list = Float64List.fromList(other._list);
      return DartList.fromList(list);
    } else if (other is CList) {
      final list = other._mem.asTypedList(other.length);
      return DartList.fromList(list);
    }
    final cSrc = other.read();
    try {
      final list = cSrc._mem.asTypedList(cSrc.length);
      return DartList.fromList(list);
    } finally {
      cSrc.release();
    }
  }

  @override
  DeviceType get deviceType => DeviceType.dart;

  @override
  int get deviceId => 0;

  @override
  int get length => _list.length;

  @override
  int get lengthBytes => length * 8;

  @override
  double operator [](int index) => _list[index];

  @override
  void operator []=(int index, double value) {
    _list[index] = value;
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => ffi.nullptr;

  @override
  void release() {}

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      _list.setAll(0, src._mem.asTypedList(length));
      return;
    } else if (src is DartList) {
      _list.setAll(0, src._list);
      return;
    }
    final cSrc = src.read();
    try {
      _list.setAll(0, cSrc._mem.asTypedList(cSrc.length));
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      dst._mem.asTypedList(dst.length).setAll(0, _list);
      return;
    } else if (dst is DartList) {
      dst._list.setAll(0, _list);
      return;
    }
    final cSrc = read();
    try {
      dst.copyFrom(cSrc);
    } finally {
      cSrc.release();
    }
  }

  @override
  CList read() {
    final clist = CList.allocate(_list.length);
    clist._mem.asTypedList(_list.length).setAll(0, _list);
    return clist;
  }
}

class CList extends NList {
  ffi.Pointer<ffi.Double> _mem;

  int _length;

  CList._(this._mem, this._length);

  static CList copy(NList other) {
    final clist = CList.allocate(other.length);
    clist.copyFrom(other);
    return clist;
  }

  static CList fromList(List<double> list) {
    final clist = CList.allocate(list.length);
    clist._mem.asTypedList(list.length).setAll(0, list);
    return clist;
  }

  static CList allocate(int length) {
    final mem = ffi.calloc<ffi.Double>(length * 8);
    return CList._(mem, length);
  }

  @override
  DeviceType get deviceType => DeviceType.c;

  @override
  int get deviceId => 0;

  @override
  int get length => _length;

  @override
  int get lengthBytes => length * 8;

  @override
  double operator [](int index) => _mem[index];

  @override
  void operator []=(int index, double value) {
    _mem[index] = value;
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _mem;

  void resize(int length) {
    final newPtr = CListFFIFunctions.realloc(_mem.cast(), length * 8);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    _length = length;
  }

  @override
  void release() {
    ffi.malloc.free(_mem);
  }

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CListFFIFunctions.memcpy(_mem.cast(), src._mem.cast(), lengthBytes);
      return;
    } else if (src is DartList) {
      _mem.asTypedList(length).setAll(0, src._list);
      return;
    }
    src.copyTo(this);
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      CListFFIFunctions.memcpy(dst._mem.cast(), _mem.cast(), lengthBytes);
      return;
    } else if (dst is DartList) {
      dst._list.setAll(0, _mem.asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }

  @override
  CList read() {
    final clist = CList.allocate(length);
    CListFFIFunctions.memcpy(clist._mem.cast(), _mem.cast(), lengthBytes);
    return clist;
  }
}

abstract class CListFFIFunctions {
  static late final ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void> oldPtr, int size) realloc;
  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  static void initialize(ffi.DynamicLibrary dylib) {
    realloc = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int)>('realloc');
    memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('memcpy');
  }
}

class CudaList extends NList {
  final ffi.Pointer<ffi.Double> _mem;
  @override
  final int length;
  final int _deviceId;

  CudaList._(this._mem, this.length, this._deviceId);

  static CudaList fromList(List<double> list, {int deviceId = 0}) {
    final clist = CudaList.allocate(list.length, deviceId: deviceId);
    clist.copyFrom(DartList.fromList(list));
    return clist;
  }

  static CudaList copy(NList other, {int deviceId = 0}) {
    final clist = CudaList.allocate(other.length, deviceId: deviceId);
    clist.copyFrom(other);
    return clist;
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
    ffi.Pointer<ffi.Double> d = ffi.calloc.allocate(byteSize);
    try {
      CudaFFIFunctions.memcpy(d.cast(), (_mem + index).cast(), byteSize);
      return d.value;
    } finally {
      ffi.calloc.free(d);
    }
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    ffi.Pointer<ffi.Double> d = ffi.calloc.allocate(byteSize);
    try {
      d.value = value;
      CudaFFIFunctions.memcpy((_mem + index * 8).cast(), d.cast(), byteSize);
    } finally {
      ffi.calloc.free(d);
    }
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _mem;

  static CudaList allocate(int length, {int deviceId = 0}) {
    final mem = CudaFFIFunctions.allocate(length, deviceId);
    return CudaList._(mem.cast(), length, deviceId);
  }

  @override
  void release() => CudaFFIFunctions.release(_mem.cast());

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CudaFFIFunctions.memcpy(_mem.cast(), src._mem.cast(), lengthBytes);
      return;
    }
    final cSrc = src.read();
    try {
      CudaFFIFunctions.memcpy(_mem.cast(), cSrc._mem.cast(), cSrc.length);
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(NList dst) {
    if (dst is CList) {
      CudaFFIFunctions.memcpy(dst._mem.cast(), dst._mem.cast(), dst.length);
      return;
    }
    final cSrc = read();
    try {
      dst.copyFrom(dst);
    } finally {
      cSrc.release();
    }
  }

  @override
  CList read() {
    final clist = CList.allocate(length);
    CudaFFIFunctions.memcpy(clist._mem.cast(), _mem.cast(), clist.lengthBytes);
    return clist;
  }

  static const byteSize = 8;
}

abstract class CudaFFIFunctions {
  static void initialize(ffi.DynamicLibrary dylib) {
    allocate = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Uint64, ffi.Int32),
        ffi.Pointer<ffi.Void> Function(int, int)>('allocate');
    release = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('release');
    memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('memcpy');
  }

  static late final ffi.Pointer<ffi.Void> Function(int size, int device)
      allocate;
  static late final void Function(ffi.Pointer<ffi.Void> ptr) release;

  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  static late final void Function(
      ffi.Pointer<ffi.Double> out,
      ffi.Pointer<ffi.Double> inp,
      CSize2D kernS,
      CSize2D outS,
      CSize2D inS,
      CSize2D stride,
      CSize2D dialation,
      CSize2D padding,
      double padValue,
      int padMode) _maxpool2D;

  static void maxpool2D(Tensor out, Tensor inp, Size2D kernS,
      {Size2D stride = const Size2D(rows: 1, cols: 1),
      Size2D padding = const Size2D(rows: 0, cols: 0),
      double padValue = 0,
      PadMode padMode = PadMode.constant,
      Size2D dilation = const Size2D(rows: 1, cols: 1)}) {
    // TODO transfer to device if necessary instead?
    if (out.deviceType != DeviceType.cuda) {
      throw ArgumentError('Output tensor must be on CUDA device');
    }
    if (out.deviceType != inp.deviceType) {
      inp = inp.to(out.deviceType, deviceId: out.deviceId);
      // TODO release this after usage
    }

    final arena = ffi.Arena();
    try {
      final kernSPtr = CSize2D.fromSize2D(kernS, allocator: arena);
      final outSPtr = CSize2D.fromSize2D(out.size.twoD, allocator: arena);
      final inSPtr = CSize2D.fromSize2D(inp.size.twoD, allocator: arena);
      final strideSPtr = CSize2D.fromSize2D(stride, allocator: arena);
      final dilationSPtr = CSize2D.fromSize2D(dilation, allocator: arena);
      final paddingSPtr = CSize2D.fromSize2D(padding, allocator: arena);

      _maxpool2D(
          out.ptr,
          inp.ptr,
          kernSPtr.ref,
          outSPtr.ref,
          inSPtr.ref,
          strideSPtr.ref,
          dilationSPtr.ref,
          paddingSPtr.ref,
          padValue,
          padMode.index);
    } finally {
      arena.releaseAll();
    }
  }
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

final class CSize2D extends ffi.Struct {
  @ffi.Int32()
  external int r;

  @ffi.Int32()
  external int c;

  static ffi.Pointer<CSize2D> fromSize2D(Size2D size,
      {ffi.Allocator allocator = ffi.malloc}) {
    final cSize = allocator.allocate<CSize2D>(ffi.sizeOf<CSize2D>());
    cSize.ref.r = size.rows;
    cSize.ref.c = size.cols;
    return cSize;
  }
}

enum PadMode { constant, reflect, replicate, circular }
