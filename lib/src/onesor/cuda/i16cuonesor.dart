part of 'cuonesor.dart';

abstract mixin class I16CuOnesor implements CuOnesor<int>, I16Onesor {
  @override
  ffi.Pointer<ffi.Int16> get ptr;

  static I16CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _I16CuOnesor.sized(stream, length, context: context);

  static I16CuOnesor fromList(CudaStream stream, Int16List list,
          {Context? context}) =>
      _I16CuOnesor.fromList(stream, list, context: context);

  static I16CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _I16CuOnesor.copy(other, stream: stream, context: context);

  @override
  I16COnesor read({Context? context, CudaStream? stream}) {
    final ret = I16COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  I16CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = I16CuOnesor.sized(stream, length, context: context);
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
  I16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16CuOnesorView(this, start, length);
  }
}

class _I16CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        I16Onesor,
        CuOnesor<int>,
        _CuOnesorMixin<int>,
        I16CuOnesor
    implements I16CuOnesor {
  @override
  final CuPtr<ffi.Int16> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _I16CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _I16CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr =
        CuPtr<ffi.Int16>.allocate(stream, length * Int16List.bytesPerElement);
    return _I16CuOnesor(ptr, length, stream.deviceId, context: context);
  }

  static _I16CuOnesor fromList(CudaStream stream, Int16List list,
      {Context? context}) {
    final ret = _I16CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(I16DartOnesor(list), stream: stream);
    return ret;
  }

  static _I16CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _I16CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Int16> get ptr => _ptr.ptr;
}

class I16CuOnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        I16Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        _CuOnesorViewMixin<int>,
        I16CuOnesor
    implements I16CuOnesor, CuOnesorView<int>, I16OnesorView {
  @override
  final I16CuOnesor _inner;

  @override
  final int offset;

  @override
  final int length;

  I16CuOnesorView(this._inner, this.offset, this.length);

  @override
  int get deviceId => _inner.deviceId;

  @override
  late final ffi.Pointer<ffi.Int16> ptr = _inner.ptr.cast<ffi.Int16>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  I16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16CuOnesorView(_inner, start + offset, length);
  }
}
