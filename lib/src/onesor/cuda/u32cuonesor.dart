part of 'cuonesor.dart';

abstract mixin class U32CuOnesor implements CuOnesor<int>, U32Onesor {
  @override
  ffi.Pointer<ffi.Uint32> get ptr;

  static U32CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U32CuOnesor.sized(stream, length, context: context);

  static U32CuOnesor fromList(CudaStream stream, Uint32List list,
          {Context? context}) =>
      _U32CuOnesor.fromList(stream, list, context: context);

  static U32CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U32CuOnesor.copy(other, stream: stream, context: context);

  @override
  U32COnesor read({Context? context, CudaStream? stream}) {
    stream = stream ?? CudaStream.noStream(deviceId);
    final ret = U32COnesor.sized(length, context: context);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  U32CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = U32CuOnesor.sized(stream, length, context: context);
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
  U32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32CuOnesorView(this, start, length);
  }
}

class _U32CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        U32Onesor,
        CuOnesor<int>,
        _CuOnesorMixin<int>,
        U32CuOnesor
    implements U32CuOnesor {
  @override
  final CuPtr<ffi.Uint32> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U32CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U32CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = CuPtr<ffi.Uint32>.allocate(stream, length * Uint32List.bytesPerElement);
    return _U32CuOnesor(ptr, length, stream.deviceId, context: context);
  }

  static _U32CuOnesor fromList(CudaStream stream, Uint32List list,
      {Context? context}) {
    final ret = _U32CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U32DartOnesor(list), stream: stream);
    return ret;
  }

  static _U32CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    // TODO simplify with link resource
    try {
      stream = stream ?? CudaStream.noStream(other.deviceId);
      final ret = _U32CuOnesor.sized(stream, other.length, context: context);
      lContext.releaseOnErr(ret);
      ret.copyFrom(other, stream: stream);
      return ret;
    } catch (e) {
      lContext.release(isError: true);
      rethrow;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Uint32> get ptr => _ptr.ptr;
}

class U32CuOnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        U32Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        _CuOnesorViewMixin<int>,
        U32CuOnesor
    implements U32CuOnesor, CuOnesorView<int>, U32OnesorView {
  @override
  final U32CuOnesor _inner;

  @override
  final int offset;

  @override
  final int length;

  U32CuOnesorView(this._inner, this.offset, this.length);

  @override
  int get deviceId => _inner.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint32> ptr = _inner.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32CuOnesorView(_inner, start + offset, length);
  }
}
