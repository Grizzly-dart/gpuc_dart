import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'dart:typed_data';

abstract class NList {
  DeviceType get device;

  int get deviceId;

  int get length;

  int get lengthBytes;

  double operator [](int index);

  void operator []=(int index, double value);

  void release();

  // TODO implement partial write
  void copyFrom(NList src);

  // TODO implement partial read
  void copyTo(NList dst);

  CList read();

  static NList allocate(int length,
      {DeviceType device = DeviceType.c, int deviceId = 0}) {
    switch (device) {
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

class DartList implements NList {
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
  DeviceType get device => DeviceType.dart;

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

class CList implements NList {
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
  DeviceType get device => DeviceType.c;

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

  void resize(int length) {
    final newPtr = CListFFIFunctions.realloc(_mem, length * 8);
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
      ffi.Pointer<void> oldPtr, int size) realloc;
  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;
}

class CudaList implements NList {
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
  DeviceType get device => DeviceType.cuda;

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
  static late final ffi.Pointer<ffi.Void> Function(int size, int device)
      allocate;
  static late final void Function(ffi.Pointer<ffi.Void> ptr) release;

  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  static late final void Function(
      ffi.Pointer<ffi.Double> out,
      ffi.Pointer<ffi.Double> inp,
      int kernSX,
      int kernSY,
      int outSX,
      int outSY,
      int inpSX,
      int inpSY,
      int padSX,
      int padSY,
      double padValue,
      int padMode,
      int dilationSX,
      int dilationSY) _maxpool2D;

  static void maxpool2D(CudaList output, CudaList input, CSize2D kernelSize) {
    final ffi.Pointer<CSize2D> kernelSizePtr = ffi.malloc.allocate();
    kernelSizePtr.ref;
    _maxpool2D(output._mem, input._mem, kernelSize.rows, kernelSize.cols, inputSize.rows, inputSize.cols, inputSize.rows, inputSize.cols, 0, 0, 0, 0, 1, 1);
  }
}

enum DeviceType { c, dart, cuda, rocm, sycl }

final class CSize2D extends ffi.Struct {
  @ffi.Int32()
  external int rows;

  @ffi.Int32()
  external int cols;
}
