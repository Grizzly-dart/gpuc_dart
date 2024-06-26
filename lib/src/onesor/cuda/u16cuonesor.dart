part of 'cuonesor.dart';

abstract mixin class U16CuOnesor implements CuOnesor<int>, U16Onesor {
  @override
  ffi.Pointer<ffi.Uint16> get ptr;

  static U16CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U16CuOnesor.sized(stream, length, context: context);

  static U16CuOnesor fromList(CudaStream stream, Uint16List list,
          {Context? context}) =>
      _U16CuOnesor.fromList(stream, list, context: context);

  static U16CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U16CuOnesor.copy(other, stream: stream, context: context);

  @override
  U16COnesor read({Context? context, CudaStream? stream}) {
    final ret = U16COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  U16CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = U16CuOnesor.sized(stream, length, context: context);
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
  U16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U16CuOnesorView(this, start, length);
  }
}

class _U16CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        U16Onesor,
        CuOnesor<int>,
        _CuOnesorMixin<int>,
        U16CuOnesor
    implements U16CuOnesor {
  @override
  final CuPtr<ffi.Uint16> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U16CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U16CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr =
        CuPtr<ffi.Uint16>.allocate(stream, length * Uint16List.bytesPerElement);
    return _U16CuOnesor(ptr, length, stream.deviceId, context: context);
  }

  static _U16CuOnesor fromList(CudaStream stream, Uint16List list,
      {Context? context}) {
    final ret = _U16CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U16DartOnesor(list), stream: stream);
    return ret;
  }

  static _U16CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _U16CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Uint16> get ptr => _ptr.ptr;
}

class U16CuOnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        U16Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        _CuOnesorViewMixin<int>,
        U16CuOnesor
    implements U16CuOnesor, CuOnesorView<int>, U16OnesorView {
  @override
  final U16CuOnesor _inner;

  @override
  final int offset;

  @override
  final int length;

  U16CuOnesorView(this._inner, this.offset, this.length);

  @override
  int get deviceId => _inner.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint16> ptr = _inner.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U16CuOnesorView(_inner, start + offset, length);
  }
}
