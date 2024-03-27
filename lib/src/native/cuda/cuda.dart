import 'dart:async';
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
    return; // CUDA not supported for MacOSX
  } else if (Platform.isWindows) {
    os = 'windows';
    return; // TODO windows not supported yet
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

  bool exists() {
    if (_cuda == null && CudaFFI.instance == null) {
      return false;
    }
    return true;
  }

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

  void sin(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.sin(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void cos(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.cos(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void tan(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.tan(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sinh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.sinh(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void cosh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.cosh(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void tanh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuda.tanh(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void transpose2D(CudaStream stream, F64Ptr out, F64Ptr inp, Dim3 size) {
    final ctx = Context();
    try {
      final sizePtr = CDim3.from(size);
      final err = cuda.transpose2d(stream.ptr, out, inp, sizePtr.ptr.ref);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void pickRows(CudaStream stream, ffi.Pointer out, ffi.Pointer inp,
      ffi.Pointer indices, Dim2 size) {
    final type = NumType.typeOf(out); // TODO fix this
    final iType = NumType.typeOf(indices); // TODO fix this

    final sizePtr = CDim2.from(size);
    final err = cuda.pickRows(stream.ptr, out.cast(), inp.cast(),
        indices.cast(), sizePtr.ptr.ref, type.id, iType.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void addition(
      CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size) {
    final key = '${out.type.short}_${inp1.type.short}_${inp2.type.short}';
    final err =
        cuda.additions[key]!(stream.ptr, out.ptr, inp1.ptr, inp2.ptr, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sub(CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size) {
    final key = '${out.type.short}_${inp1.type.short}_${inp2.type.short}';
    final err = cuda.subs[key]!(stream.ptr, out.ptr, inp1.ptr, inp2.ptr, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void mul(CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size) {
    final key = '${out.type.short}_${inp1.type.short}_${inp2.type.short}';
    final err = cuda.muls[key]!(stream.ptr, out.ptr, inp1.ptr, inp2.ptr, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void div(CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size) {
    final key = '${out.type.short}_${inp1.type.short}_${inp2.type.short}';
    final err = cuda.divs[key]!(stream.ptr, out.ptr, inp1.ptr, inp2.ptr, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sum2d(CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuda.sum2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
          out.type.id, inp.type.id);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void mean2d(CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuda.mean2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
          out.type.id, inp.type.id);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void variance2d(
      CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS, int correction) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuda.variance2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
          correction, 0, out.type.id, inp.type.id);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void std2d(
      CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS, int correction) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuda.variance2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
          correction, 0xFF, out.type.id, inp.type.id);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void normalize2d(
      CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS, double epsilon) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuda.normalize2d(stream.ptr, out.ptr, inp.ptr,
          sizePtr.ptr.ref, epsilon, out.type.id, inp.type.id);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
    } finally {
      ctx.release();
    }
  }

  void matmul(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2, int m,
      int n, int k, int batches) {
    final err = cuda.matmul(stream.ptr, out, inp1, inp2, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulT(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2, int m,
      int n, int k, int batches) {
    final err = cuda.matmulT(stream.ptr, out, inp1, inp2, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulCadd(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2,
      F64Ptr add, int m, int n, int k, int batches) {
    final err =
        cuda.matmulCadd(stream.ptr, out, inp1, inp2, add, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulTCadd(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2,
      F64Ptr add, int m, int n, int k, int batches) {
    final err =
        cuda.matmulTCadd(stream.ptr, out, inp1, inp2, add, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void maxPool2D(CudaStream stream, ffi.Pointer out, ffi.Pointer inp,
      {required Dim2 kernSize,
      required Dim2 outSize,
      required Dim2 inpSize,
      required int matrices,
      Dim2 stride = const Dim2(1, 1),
      Dim2 padding = const Dim2(0, 0),
      Dim2 dilation = const Dim2(1, 1)}) {
    final kernSPtr = CDim2.from(kernSize);
    final outSPtr = CDim2.from(outSize);
    final inSPtr = CDim2.from(inpSize);
    final strideSPtr = CDim2.from(stride);
    final dilationSPtr = CDim2.from(dilation);
    final paddingSPtr = CDim2.from(padding);

    final err = cuda.maxPool2D(
        stream.ptr,
        out.cast(),
        inp.cast(),
        kernSPtr.ptr.ref,
        outSPtr.ptr.ref,
        inSPtr.ptr.ref,
        matrices,
        paddingSPtr.ptr.ref,
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
    final outSC = CDim3.from(outS);
    final inpSC = CDim3.from(inpS);
    final kernSC = CDim2.from(kernS);
    final paddingSC = CDim2.from(padding);
    final strideSC = CDim2.from(stride);
    final dilationSC = CDim2.from(dilation);
    final err = cuda.conv2D(
        stream.ptr,
        out,
        inp,
        kernel,
        batches,
        outSC.ptr.ref,
        inpSC.ptr.ref,
        kernSC.ptr.ref,
        groups,
        paddingSC.ptr.ref,
        padMode.index,
        pad,
        strideSC.ptr.ref,
        dilationSC.ptr.ref);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void eluActivation(
      CudaStream stream, NumPtr out, NumPtr inp, int size, double alpha) {
    final err = cuda.eluActivation(
        stream.ptr, out.ptr, inp.ptr, size, alpha, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sigmoidActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err =
        cuda.sigmoidActivation(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void siluActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err =
        cuda.siluActivation(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void softplusActivation(CudaStream stream, NumPtr out, NumPtr inp, int size,
      int beta, int threshold) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err = cuda.softplusActivation(
        stream.ptr, out.ptr, inp.ptr, size, beta, threshold, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void softsignActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err = cuda.softsignActivation(
        stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void mishActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err =
        cuda.mishActivation(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void minThreshold(CudaStream stream, NumPtr out, NumPtr inp, num threshold,
      num value, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final thresholdPtr = inp.type.allocateForValue(threshold);
    final valuePtr = inp.type.allocateForValue(value);
    final err = cuda.minThreshold(stream.ptr, out.ptr, inp.ptr,
        thresholdPtr.ptr, valuePtr.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
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

  Future<void> sync() async {
    final completer = Completer<void>();
    void callback(StrPtr err) {
      if (err != ffi.nullptr) {
        completer.completeError(CudaException(err.toDartString()));
      } else {
        completer.complete();
      }
    }

    final nc = ffi.NativeCallable<ffi.Void Function(StrPtr)>.listener(callback);
    try {
      final err = CudaFFI.instance!.syncStream(_stream, nc.nativeFunction);
      if (err != ffi.nullptr) {
        completer.completeError(CudaException(err.toDartString()));
      }
      await completer.future;
    } finally {
      nc.close();
    }
  }

  @override
  void release() {
    if (_stream == ffi.nullptr) return;

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
    cffi!.finalizer.attach(ret, ptr.cast());
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
    cffi!.finalizer.attach(ret, ret.ptr.cast());
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

extension CudaSetGet on Cuda {
  double getF64(ffi.Pointer<ffi.Double> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Double>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Double>());
      return dst.ptr.cast<ffi.Double>().value;
    } finally {
      context.release();
    }
  }

  void setF64(
      ffi.Pointer<ffi.Double> dst, int index, double value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Double>());
      src.ptr.cast<ffi.Double>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Double>());
    } finally {
      context.release();
    }
  }

  double getF32(ffi.Pointer<ffi.Float> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Float>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Float>());
      return dst.ptr.cast<ffi.Float>().value;
    } finally {
      context.release();
    }
  }

  void setF32(ffi.Pointer<ffi.Float> dst, int index, double value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Float>());
      src.ptr.cast<ffi.Float>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Float>());
    } finally {
      context.release();
    }
  }

  int getU64(ffi.Pointer<ffi.Uint64> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Uint64>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Uint64>());
      return dst.ptr.cast<ffi.Uint64>().value;
    } finally {
      context.release();
    }
  }

  void setU64(ffi.Pointer<ffi.Uint64> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Uint64>());
      src.ptr.cast<ffi.Uint64>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Uint64>());
    } finally {
      context.release();
    }
  }

  int getU32(ffi.Pointer<ffi.Uint32> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Uint32>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Uint32>());
      return dst.ptr.cast<ffi.Uint32>().value;
    } finally {
      context.release();
    }
  }

  void setU32(ffi.Pointer<ffi.Uint32> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Uint32>());
      src.ptr.cast<ffi.Uint32>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Uint32>());
    } finally {
      context.release();
    }
  }

  int getU16(ffi.Pointer<ffi.Uint16> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Uint16>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Uint16>());
      return dst.ptr.cast<ffi.Uint16>().value;
    } finally {
      context.release();
    }
  }

  void setU16(ffi.Pointer<ffi.Uint16> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Uint16>());
      src.ptr.cast<ffi.Uint16>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Uint16>());
    } finally {
      context.release();
    }
  }

  int getU8(ffi.Pointer<ffi.Uint8> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Uint8>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Uint8>());
      return dst.ptr.cast<ffi.Uint8>().value;
    } finally {
      context.release();
    }
  }

  void setU8(ffi.Pointer<ffi.Uint8> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Uint8>());
      src.ptr.cast<ffi.Uint8>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Uint8>());
    } finally {
      context.release();
    }
  }

  int getI64(ffi.Pointer<ffi.Int64> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Int64>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Int64>());
      return dst.ptr.cast<ffi.Int64>().value;
    } finally {
      context.release();
    }
  }

  void setI64(ffi.Pointer<ffi.Int64> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Int64>());
      src.ptr.cast<ffi.Int64>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Int64>());
    } finally {
      context.release();
    }
  }

  int getI32(ffi.Pointer<ffi.Int32> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Int32>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Int32>());
      return dst.ptr.cast<ffi.Int32>().value;
    } finally {
      context.release();
    }
  }

  void setI32(ffi.Pointer<ffi.Int32> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Int32>());
      src.ptr.cast<ffi.Int32>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Int32>());
    } finally {
      context.release();
    }
  }

  int getI16(ffi.Pointer<ffi.Int16> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Int16>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(stream, dst.ptr.cast(), (src + index).cast(),
          ffi.sizeOf<ffi.Int16>());
      return dst.ptr.cast<ffi.Int16>().value;
    } finally {
      context.release();
    }
  }

  void setI16(ffi.Pointer<ffi.Int16> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Int16>());
      src.ptr.cast<ffi.Int16>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(stream, (dst + index).cast(), src.ptr.cast(),
          ffi.sizeOf<ffi.Int16>());
    } finally {
      context.release();
    }
  }

  int getI8(ffi.Pointer<ffi.Int8> src, int index, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final dst = CPtr.allocate(ffi.sizeOf<ffi.Int8>());
      stream = stream ?? CudaStream(deviceId, context: context);
      memcpy(
          stream, dst.ptr.cast(), (src + index).cast(), ffi.sizeOf<ffi.Int8>());
      return dst.ptr.cast<ffi.Int8>().value;
    } finally {
      context.release();
    }
  }

  void setI8(ffi.Pointer<ffi.Int8> dst, int index, int value, int deviceId,
      {CudaStream? stream}) {
    final context = Context();
    try {
      final src = CPtr.allocate(ffi.sizeOf<ffi.Int8>());
      src.ptr.cast<ffi.Int8>().value = value;
      stream = stream ?? CudaStream(deviceId, context: context);
      cuda.memcpy(
          stream, (dst + index).cast(), src.ptr.cast(), ffi.sizeOf<ffi.Int8>());
    } finally {
      context.release();
    }
  }
}
