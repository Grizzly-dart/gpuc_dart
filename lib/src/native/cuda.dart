import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

typedef Op1D2Inp = ffi.Pointer<ffi.Utf8> Function(
    ffi.Pointer<CCudaStream> stream,
    ffi.Pointer<ffi.Void> out,
    ffi.Pointer<ffi.Void> inp1,
    ffi.Pointer<ffi.Void>,
    int size);
typedef Op1D2InpNative = ffi.Pointer<ffi.Utf8> Function(
    ffi.Pointer<CCudaStream> stream,
    ffi.Pointer<ffi.Void> out,
    ffi.Pointer<ffi.Void> inp1,
    ffi.Pointer<ffi.Void>,
    ffi.Uint32 size);

typedef Op2D = ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream> stream,
    ffi.Pointer<ffi.Void> out, ffi.Pointer<ffi.Void> inp1, CSize2D);
typedef Op2DNative = ffi.Pointer<ffi.Utf8> Function(
    ffi.Pointer<CCudaStream> stream,
    ffi.Pointer<ffi.Void> out,
    ffi.Pointer<ffi.Void> inp1,
    CSize2D);

abstract class CudaFFIFunctions {
  static void initialize(ffi.DynamicLibrary dylib) {
    CCudaStream.initializeLib(dylib);

    _getDeviceProps = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<_CudaDeviceProps>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<_CudaDeviceProps>, int)>('libtcCudaGetDeviceProps');
    _getMemInfo = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<_CudaMemInfo>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<_CudaMemInfo>, int)>('libtcCudaGetMemInfo');

    allocate = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, int)>('libtcCudaAlloc');
    release = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>)>('libtcCudaFree');
    _memcpy = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>,
            ffi.Pointer<ffi.Void>,
            int)>('libtcCudaMemcpy');

    _addition =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaAddCkern');
    sum2D = dylib.lookupFunction<Op2DNative, Op2D>('libtcCudaSum2DCkern');
  }

  static late final ffi.Pointer<ffi.Utf8> Function(
      ffi.Pointer<_CudaDeviceProps>, int device) _getDeviceProps;
  static late final ffi.Pointer<ffi.Utf8> Function(
      ffi.Pointer<_CudaMemInfo>, int device) _getMemInfo;

  static late final ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
      ffi.Pointer<ffi.Pointer<ffi.Void>>, int size) allocate;
  static late final ffi.Pointer<ffi.Utf8> Function(
      ffi.Pointer<CCudaStream> ptr, ffi.Pointer<ffi.Void>) release;
  static late final ffi.Pointer<ffi.Utf8> Function(
      ffi.Pointer<CCudaStream> stream,
      ffi.Pointer<ffi.Void> dst,
      ffi.Pointer<ffi.Void> src,
      int size) _memcpy;

  static late final Op1D2Inp _addition;
  static late final Op2D sum2D;

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

  static CudaDeviceProps getDeviceProps(int device) {
    final ptr =
        ffi.calloc.allocate<_CudaDeviceProps>(ffi.sizeOf<_CudaDeviceProps>());
    final ret = CudaDeviceProps(ptr);
    CListFFIFunctions.finalizer.attach(ret, ptr.cast());
    final err = _getDeviceProps(ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  static CudaMemInfo getMemInfo(int device) {
    final ptr = ffi.calloc.allocate<_CudaMemInfo>(ffi.sizeOf<_CudaMemInfo>());
    final ret = CudaMemInfo(ptr);
    CListFFIFunctions.finalizer.attach(ret, ptr.cast());
    final err = _getMemInfo(ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  static void memcpy(CudaStream stream, ffi.Pointer<ffi.Void> dst,
      ffi.Pointer<ffi.Void> src, int size) {
    final err = _memcpy(stream.ptr, dst, src, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  static double getDouble(ffi.Pointer<ffi.Double> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CF64Ptr.allocate(context: context);
      stream = stream ?? CudaStream(deviceId, context: context);
      CudaFFIFunctions.memcpy(stream, dst.ptr.cast(), (src + index).cast(), 8);
      return dst.value;
    } finally {
      context.release();
    }
  }

  static double setDouble(
      ffi.Pointer<ffi.Double> dst, int index, double value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CF64Ptr.allocate(context: context);
      src.value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      CudaFFIFunctions.memcpy(stream, (dst + index).cast(), src.ptr.cast(), 8);
      return dst.value;
    } finally {
      context.release();
    }
  }

  static void addition(CudaStream stream, ffi.Pointer<ffi.Void> out,
      ffi.Pointer<ffi.Void> inp1, ffi.Pointer<ffi.Void> inp2, int size) {
    final err = _addition(stream.ptr, out, inp1, inp2, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

/*
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
   */
}

final class _CudaDeviceProps extends ffi.Struct {
  @ffi.Uint64()
  external int get totalGlobalMem;

  @ffi.Uint64()
  external int get totalConstMem;

  @ffi.Uint64()
  external int get sharedMemPerBlock;

  @ffi.Uint64()
  external int get reservedSharedMemPerBlock;

  @ffi.Uint64()
  external int get sharedMemPerMultiProcessor;

  @ffi.Uint32()
  external int get warpSize;

  @ffi.Uint32()
  external int get multiProcessorCount;

  @ffi.Uint32()
  external int get maxThreadsPerMultiProcessor;

  @ffi.Uint32()
  external int get maxThreadsPerBlock;

  @ffi.Uint32()
  external int get maxBlocksPerMultiProcessor;

  @ffi.Uint32()
  external int get l2CacheSize;

  @ffi.Uint64()
  external int get memPitch;

  @ffi.Uint32()
  external int get memoryBusWidth;

  @ffi.Uint32()
  external int get pciBusID;

  @ffi.Uint32()
  external int get pciDeviceID;

  @ffi.Uint32()
  external int get pciDomainID;
}

class CudaDeviceProps implements ffi.Finalizable {
  final ffi.Pointer<_CudaDeviceProps> _ptr;

  CudaDeviceProps(this._ptr);

  _CudaDeviceProps get _props => _ptr.ref;

  int get totalGlobalMem => _props.totalGlobalMem;

  int get totalConstMem => _props.totalConstMem;

  int get sharedMemPerBlock => _props.sharedMemPerBlock;

  int get reservedSharedMemPerBlock => _props.reservedSharedMemPerBlock;

  int get sharedMemPerMultiProcessor => _props.sharedMemPerMultiProcessor;

  int get warpSize => _props.warpSize;

  int get multiProcessorCount => _props.multiProcessorCount;

  int get maxThreadsPerMultiProcessor => _props.maxThreadsPerMultiProcessor;

  int get maxThreadsPerBlock => _props.maxThreadsPerBlock;

  int get maxBlocksPerMultiProcessor => _props.maxBlocksPerMultiProcessor;

  int get l2CacheSize => _props.l2CacheSize;

  int get memPitch => _props.memPitch;

  int get memoryBusWidth => _props.memoryBusWidth;

  int get pciBusID => _props.pciBusID;

  int get pciDeviceID => _props.pciDeviceID;

  int get pciDomainID => _props.pciDomainID;

  Map<String, dynamic> toJson() => {
        'totalGlobalMem': totalGlobalMem,
        'totalConstMem': totalConstMem,
        'sharedMemPerBlock': sharedMemPerBlock,
        'reservedSharedMemPerBlock': reservedSharedMemPerBlock,
        'sharedMemPerMultiProcessor': sharedMemPerMultiProcessor,
        'warpSize': warpSize,
        'multiProcessorCount': multiProcessorCount,
        'maxThreadsPerMultiProcessor': maxThreadsPerMultiProcessor,
        'maxThreadsPerBlock': maxThreadsPerBlock,
        'maxBlocksPerMultiProcessor': maxBlocksPerMultiProcessor,
        'l2CacheSize': l2CacheSize,
        'memPitch': memPitch,
        'memoryBusWidth': memoryBusWidth,
        'pciBusID': pciBusID,
        'pciDeviceID': pciDeviceID,
        'pciDomainID': pciDomainID,
      };

  @override
  String toString() => toJson().toString();
}

final class CCudaStream extends ffi.Struct {
  external final ffi.Pointer<ffi.Void> stream;
  @ffi.Int32()
  external final int deviceId;

  Map<String, dynamic> toJson() => {
        'stream': stream,
        'deviceId': deviceId,
      };

  @override
  String toString() => toJson().toString();

  static ffi.Pointer<CCudaStream> create(int device) {
    final stream = ffi.calloc.allocate<CCudaStream>(ffi.sizeOf<CCudaStream>());
    try {
      final err = _create(stream, device);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
      return stream;
    } catch (e) {
      ffi.calloc.free(stream);
      rethrow;
    }
  }

  static void initializeLib(ffi.DynamicLibrary dylib) {
    _create = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, int)>('libtcCudaCreateStream');
    _destroy = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>)>('libtcCudaDestroyStream');
  }
}

late final ffi.Pointer<ffi.Utf8> Function(
    ffi.Pointer<CCudaStream> stream, int device) _create;
late final ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream> stream)
    _destroy;

class CudaStream extends Resource {
  ffi.Pointer<CCudaStream> _stream;

  CudaStream._(this._stream, {Context? context}) {
    context?.add(this);
  }

  factory CudaStream(int device, {Context? context}) {
    final stream = CCudaStream.create(device);
    final s = CudaStream._(stream, context: context);
    return s;
  }

  ffi.Pointer<CCudaStream> get ptr => _stream;

  int get deviceId => _stream.ref.deviceId;

  @override
  void release() {
    final err = _destroy(_stream);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    ffi.malloc.free(_stream);
    _stream = ffi.nullptr;
  }
}

class CudaException implements Exception {
  final String message;

  CudaException(this.message);

  @override
  String toString() => message;
}

final class _CudaMemInfo extends ffi.Struct {
  @ffi.Uint64()
  external int get free;

  @ffi.Uint64()
  external int get total;
}

class CudaMemInfo implements ffi.Finalizable {
  final ffi.Pointer<_CudaMemInfo> _ptr;

  CudaMemInfo(this._ptr);

  _CudaMemInfo get _info => _ptr.ref;

  int get free => _info.free;

  int get total => _info.total;

  Map<String, dynamic> toJson() => {
        'free': free,
        'total': total,
      };

  @override
  String toString() => toJson().toString();

  String toHumanString() =>
      'Free: ${intToHumanReadable(free)}, Total: ${intToHumanReadable(total)}';
}

String intToHumanReadable(int value) {
  if (value < 1024) {
    return '$value B';
  }
  if (value < 1024 * 1024) {
    return '${(value / 1024).toStringAsFixed(2)} KB';
  }
  if (value < 1024 * 1024 * 1024) {
    return '${(value / (1024 * 1024)).toStringAsFixed(2)} MB';
  }
  return '${(value / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
}
