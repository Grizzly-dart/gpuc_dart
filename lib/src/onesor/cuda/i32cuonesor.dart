part of 'cuonesor.dart';

abstract mixin class I32CuOnesor implements CuOnesor<int>, I32Onesor {
  @override
  ffi.Pointer<ffi.Int32> get ptr;

  static I32CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _I32CuOnesor.sized(stream, length, context: context);

  static I32CuOnesor fromList(CudaStream stream, Int32List list,
          {Context? context}) =>
      _I32CuOnesor.fromList(stream, list, context: context);

  static I32CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _I32CuOnesor.copy(other, stream: stream, context: context);

  @override
  I32COnesor read({Context? context, CudaStream? stream}) {
    final ret = I32COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  I32CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = I32CuOnesor.sized(stream, length, context: context);
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
  I32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32CuOnesorView(this, start, length);
  }
}

class _I32CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        I32Onesor,
        CuOnesor<int>,
        _CuOnesorMixin<int>,
        I32CuOnesor
    implements I32CuOnesor {
  @override
  final CuPtr<ffi.Int32> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _I32CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _I32CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr =
        CuPtr<ffi.Int32>.allocate(stream, length * Int32List.bytesPerElement);
    return _I32CuOnesor(ptr, length, stream.deviceId, context: context);
  }

  static _I32CuOnesor fromList(CudaStream stream, Int32List list,
      {Context? context}) {
    final ret = _I32CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(I32DartOnesor(list), stream: stream);
    return ret;
  }

  static _I32CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _I32CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Int32> get ptr => _ptr.ptr;
}

class I32CuOnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        I32Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        _CuOnesorViewMixin<int>,
        I32CuOnesor
    implements I32CuOnesor, CuOnesorView<int>, I32OnesorView {
  @override
  final I32CuOnesor _inner;

  @override
  final int offset;

  @override
  final int length;

  I32CuOnesorView(this._inner, this.offset, this.length);

  @override
  int get deviceId => _inner.deviceId;

  @override
  late final ffi.Pointer<ffi.Int32> ptr = _inner.ptr.cast<ffi.Int32>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  I32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32CuOnesorView(_inner, start + offset, length);
  }
}
