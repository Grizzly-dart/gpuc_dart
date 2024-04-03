part of 'cuonesor.dart';

abstract mixin class U64CuOnesor implements CuOnesor<int>, U64Onesor {
  @override
  ffi.Pointer<ffi.Uint64> get ptr;

  static U64CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U64CuOnesor.sized(stream, length, context: context);

  static U64CuOnesor fromList(CudaStream stream, Uint64List list,
          {Context? context}) =>
      _U64CuOnesor.fromList(stream, list, context: context);

  static U64CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U64CuOnesor.copy(other, stream: stream, context: context);

  @override
  U64COnesor read({Context? context, CudaStream? stream}) {
    final ret = U64COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  U64CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId, context: lContext);
      final ret = U64CuOnesor.sized(stream, length, context: context);
      lContext.releaseOnErr(ret);
      cuda.memcpy(stream, ret.ptr.cast(), (ptr + bytesPerItem).cast(),
          length * bytesPerItem);
      return ret;
    } catch (e) {
      lContext.release(isError: true);
      rethrow;
    } finally {
      lContext.release();
    }
  }

  @override
  U64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64CuOnesorView(this, start, length);
  }
}

class _U64CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        U64Onesor,
        CuOnesor<int>,
        _CuOnesorMixin<int>,
        U64CuOnesor
    implements U64CuOnesor {
  @override
  final CuPtr<ffi.Uint64> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U64CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U64CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr =
        CuPtr<ffi.Uint64>.allocate(stream, length * Uint64List.bytesPerElement);
    return _U64CuOnesor(ptr, length, stream.deviceId, context: context);
  }

  static _U64CuOnesor fromList(CudaStream stream, Uint64List list,
      {Context? context}) {
    final ret = _U64CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U64DartOnesor(list), stream: stream);
    return ret;
  }

  static _U64CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _U64CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Uint64> get ptr => _ptr.ptr;
}

class U64CuOnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        U64Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        _CuOnesorViewMixin<int>,
        U64CuOnesor
    implements U64CuOnesor, CuOnesorView<int>, U64OnesorView {
  @override
  final U64CuOnesor _inner;

  @override
  final int offset;

  @override
  final int length;

  U64CuOnesorView(this._inner, this.offset, this.length);

  @override
  int get deviceId => _inner.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint64> ptr = _inner.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64CuOnesorView(_inner, start + offset, length);
  }
}
