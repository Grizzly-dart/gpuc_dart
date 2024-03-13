import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/cuda/native_types.dart';
import 'package:gpuc_dart/src/util/string.dart';
import 'package:path/path.dart' as path;

void initializeTensorCuda({String? libPath}) {
  String os;
  if (Platform.isLinux) {
    os = 'linux';
  } else if (Platform.isMacOS) {
    os = 'darwin';
  } else if (Platform.isWindows) {
    os = 'windows';
  } else {
    return;
  }

  String libraryPath;
  if (libPath != null) {
    libraryPath = libPath;
  } else {
    libraryPath = path.join(Directory.current.path, 'lib', 'asset', os);
  }
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libtensorcuda.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libtensorcuda.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorcuda.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CudaFFI.initialize(dylib);
}

final Cuda cuda = Cuda(null);

class Cuda {
  final CudaFFI? _cuda;

  Cuda(this._cuda);

  CudaFFI get cuda => _cuda ?? CudaFFI.instance!;

  CudaDeviceProps getDeviceProps(int device) {
    final ret = CudaDeviceProps.allocate();
    final err = cuda.getDeviceProps(ret.ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  CudaMemInfo getMemInfo(int device) {
    final ret = CudaMemInfo.allocate();
    final err = cuda.getMemInfo(ret.ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  ffi.Pointer<ffi.Void> allocate(CudaStream stream, int size) {
    final ptr = CPtr<ffi.Pointer<ffi.Void>>.allocate(
        ffi.sizeOf<ffi.Pointer<ffi.Void>>());
    final err = cuda.allocate(stream.ptr, ptr.ptr.cast(), size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ptr.ptr.value;
  }

  void memcpy(CudaStream stream, ffi.Pointer<ffi.Void> dst,
      ffi.Pointer<ffi.Void> src, int size) {
    final err = cuda.memcpy(stream.ptr, dst, src, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void memFree(CudaStream stream, ffi.Pointer<ffi.Void> ptr) {
    final err = cuda.memFree(stream.ptr, ptr);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  double getDouble(ffi.Pointer<ffi.Double> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CF64Ptr.allocate(context: context);
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(), 8);
      return dst.value;
    } finally {
      context.release();
    }
  }

  double setDouble(
      ffi.Pointer<ffi.Double> dst, int index, double value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CF64Ptr.allocate(context: context);
      src.value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream.ptr, (dst + index).cast(), src.ptr.cast(), 8);
      return dst.value;
    } finally {
      context.release();
    }
  }

  void addition(CudaStream stream, ffi.Pointer<ffi.Void> out,
      ffi.Pointer<ffi.Void> inp1, ffi.Pointer<ffi.Void> inp2, int size) {
    final err = cuda.addition(stream.ptr, out, inp1, inp2, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sum2D(CudaStream stream, ffi.Pointer<ffi.Void> out,
      ffi.Pointer<ffi.Void> inp1, Dim2 inpS) {
    final ctx = Context();
    try {
      final sizePtr = CSize2D.fromSize2D(inpS, context: ctx);
      final err = cuda.sum2D(stream.ptr, out, inp1, sizePtr.ptr.ref);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void maxPool2D(CudaStream stream, ffi.Pointer<ffi.Double> out,
      ffi.Pointer<ffi.Double> inp,
      {required Dim2 kernSize,
      required Dim2 outSize,
      required Dim2 inpSize,
      required int matrices,
      Dim2 stride = const Dim2(1, 1),
      Dim2 padding = const Dim2(0, 0),
      double pad = 0,
      PadMode padMode = PadMode.constant,
      Dim2 dilation = const Dim2(1, 1)}) {
    final kernSPtr = CSize2D.fromSize2D(kernSize);
    final outSPtr = CSize2D.fromSize2D(outSize);
    final inSPtr = CSize2D.fromSize2D(inpSize);
    final strideSPtr = CSize2D.fromSize2D(stride);
    final dilationSPtr = CSize2D.fromSize2D(dilation);
    final paddingSPtr = CSize2D.fromSize2D(padding);

    final err = cuda.maxPool2D(
        stream.ptr,
        out.cast(),
        inp.cast(),
        kernSPtr.ptr.ref,
        outSPtr.ptr.ref,
        inSPtr.ptr.ref,
        matrices,
        paddingSPtr.ptr.ref,
        padMode.index,
        pad,
        strideSPtr.ptr.ref,
        dilationSPtr.ptr.ref);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void conv2D(
      CudaStream stream,
      F64Ptr out,
      F64Ptr inp,
      F64Ptr kernel,
      int batches,
      Dim3 outS,
      Dim3 inpS,
      Dim2 kernS,
      int groups,
      Dim2 padding,
      PadMode padMode,
      double pad,
      Dim2 stride,
      Dim2 dilation) {
    // TODO
  }
}

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
    final err = CudaFFI.instance!.destroyStream(_stream);
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

class CudaMemInfo implements ffi.Finalizable {
  final ffi.Pointer<CCudaMemInfo> ptr;

  CudaMemInfo(this.ptr);

  static CudaMemInfo allocate() {
    final ptr = ffi.calloc.allocate<CCudaMemInfo>(ffi.sizeOf<CCudaMemInfo>());
    final ret = CudaMemInfo(ptr);
    CFFI.finalizer.attach(ret, ptr.cast());
    return ret;
  }

  CCudaMemInfo get _info => ptr.ref;

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

class CudaDeviceProps implements ffi.Finalizable {
  final ffi.Pointer<CCudaDeviceProps> ptr;

  CudaDeviceProps(this.ptr);

  static CudaDeviceProps allocate() {
    final ret = CudaDeviceProps(
        ffi.calloc.allocate<CCudaDeviceProps>(ffi.sizeOf<CCudaDeviceProps>()));
    CFFI.finalizer.attach(ret, ret.ptr.cast());
    return ret;
  }

  CCudaDeviceProps get _props => ptr.ref;

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
