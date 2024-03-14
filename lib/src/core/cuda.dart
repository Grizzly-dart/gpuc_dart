import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract class CudaList extends NList {
  factory CudaList.sized(CudaStream stream, int length, {Context? context}) =>
      _CudaListImpl.sized(stream, length, context: context);

  factory CudaList.fromList(CudaStream stream, List<double> list,
          {Context? context}) =>
      _CudaListImpl.fromList(stream, list, context: context);

  factory CudaList.copy(NList other, {CudaStream? stream, Context? context}) =>
      _CudaListImpl.copy(other, stream: stream, context: context);

  @override
  void copyFrom(NList src, {CudaStream? stream});

  @override
  void copyTo(NList dst, {CudaStream? stream});

  @override
  CudaList slice(int start, int length, {Context? context});

  @override
  CudaListView view(int start, int length);
}

class _CudaListImpl extends NList
    with CudaListMixin, ListMixin<double>
    implements CudaList {
  ffi.Pointer<ffi.Double> _mem;
  @override
  final int length;
  final int _deviceId;

  _CudaListImpl._(this._mem, this.length, this._deviceId, {Context? context}) {
    context?.add(this);
  }

  static _CudaListImpl sized(CudaStream stream, int length,
      {Context? context}) {
    final ptr = cuda.allocate(stream, length * NList.byteSize);
    return _CudaListImpl._(ptr.cast(), length, stream.deviceId,
        context: context);
  }

  static _CudaListImpl fromList(CudaStream stream, List<double> list,
      {Context? context}) {
    final ret = _CudaListImpl.sized(stream, list.length, context: context);
    ret.copyFrom(DartList.own(list), stream: stream);
    return ret;
  }

  static _CudaListImpl copy(NList other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret =
          _CudaListImpl.sized(stream, other.length, context: context);
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
  int get lengthBytes => length * NList.byteSize;

  @override
  double operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getDouble(_mem, index, deviceId);
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setDouble(_mem, index, value, deviceId);
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _mem;

  @override
  void release() {
    if (_mem == ffi.nullptr) return;
    final stream = CudaStream(deviceId);
    try {
      cuda.memFree(stream, _mem.cast());
      _mem = ffi.nullptr;
    } finally {
      stream.release();
    }
  }

  @override
  set length(int newLength) {
    throw UnsupportedError('Length cannot be changed');
  }
}

mixin CudaListMixin implements CudaList {
  @override
  void copyFrom(NList src, {CudaStream? stream}) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: context);
      src = src is CList ? src : src.read(context: context);
      cuda.memcpy(stream, ptr.cast(), src.ptr.cast(), lengthBytes);
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
        cuda.memcpy(stream, dst.ptr.cast(), ptr.cast(), dst.lengthBytes);
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
    final clist = CList.sized(length, context: context);
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: lContext);
      cuda.memcpy(stream, clist.ptr.cast(), ptr.cast(), clist.lengthBytes);
      return clist;
    } finally {
      lContext.release();
    }
  }

  @override
  CudaList slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      final stream = CudaStream(deviceId, context: lContext);
      final ret = CudaList.sized(stream, length, context: context);
      lContext.releaseOnErr(ret);
      cuda.memcpy(stream, ret.ptr.cast(), (ptr + start * NList.byteSize).cast(),
          length * NList.byteSize);
      return ret;
    } catch (e) {
      lContext.release(isError: true);
      rethrow;
    } finally {
      lContext.release();
    }
  }

  @override
  CudaListView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is CudaListView) {
      return CudaListView((this as CudaListView)._list,
          start + (this as CudaListView)._offset, length);
    }
    return CudaListView(this, start, length);
  }
}

class CudaListView extends NList
    with CudaListMixin, ListMixin<double>
    implements CudaList {
  final CudaList _list;
  final int _offset;
  @override
  final int length;

  CudaListView(this._list, this._offset, this.length);

  @override
  final DeviceType deviceType = DeviceType.cuda;

  @override
  final int deviceId = 0;

  @override
  double operator [](int index) => _list[_offset + index];

  @override
  void operator []=(int index, double value) => _list[_offset + index] = value;

  @override
  late final int lengthBytes = length * NList.byteSize;

  @override
  ffi.Pointer<ffi.Double> get ptr => _list.ptr + _offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of a view');
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

enum PadMode { constant, reflect, replicate, circular }
