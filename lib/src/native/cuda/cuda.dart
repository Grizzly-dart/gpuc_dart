import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/c/c_types.dart';
import 'package:gpuc_dart/src/native/cuda/cu_types.dart';
import 'package:gpuc_dart/src/util/string.dart';
import 'package:path/path.dart' as path;

export 'ffi.dart';

void initializeTensorCuda({String? libPath}) {
  // TODO support processor architecture
  String os;
  if (Platform.isLinux) {
    os = 'linux';
  } else if (Platform.isMacOS) {
    os = 'darwin';
    return; // CUDA not supported for MacOSX
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
    libraryPath = path.join(libraryPath, 'libtensorcuda.so');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorcuda.so');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CuFFI.initialize(dylib);
}

final Cuda cuda = Cuda(null);

class Cuda {
  final CuFFI? _cuFFI;

  Cuda(this._cuFFI);

  CuFFI get cuFFI => _cuFFI ?? CuFFI.instance!;

  bool exists() {
    if (_cuFFI == null && CuFFI.instance == null) {
      return false;
    }
    return true;
  }

  CudaDeviceProps getDeviceProps(int device) {
    final ret = CudaDeviceProps.allocate();
    final err = cuFFI.getDeviceProps(ret.ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  CudaMemInfo getMemInfo(int device) {
    final ret = CudaMemInfo.allocate();
    final err = cuFFI.getMemInfo(ret.ptr, device);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ret;
  }

  ffi.Pointer<ffi.Void> allocate(CudaStream stream, int size) {
    final ptr = CPtr<ffi.Pointer<ffi.Void>>.allocate(
        ffi.sizeOf<ffi.Pointer<ffi.Void>>());
    final err = cuFFI.allocate(stream.ptr, ptr.ptr.cast(), size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    return ptr.ptr.value;
  }

  void memcpy(CudaStream stream, ffi.Pointer<ffi.Void> dst,
      ffi.Pointer<ffi.Void> src, int size) {
    final err = cuFFI.memcpy(stream.ptr, dst, src, size);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void memFree(CudaStream stream, ffi.Pointer<ffi.Void> ptr) {
    final err = cuFFI.memFree(stream.ptr, ptr);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void neg(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.neg(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void binaryArith(CuOpBinary op, CudaStream stream, NumPtr out, NumPtr inp1,
      NumPtr inp2, int size) {
    final err = op(stream.ptr, out.ptr, inp1.ptr, inp2.ptr, ffi.nullptr, size,
        0, out.type.id, inp1.type.id, inp2.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void binaryArithScalar(CuOpBinary op, CudaStream stream, NumPtr out,
      NumPtr inp1, scalar, int size,
      {bool flip = false}) {
    NumType inp2Type;
    CPtr inp2 = CPtr.allocate(8);
    if (scalar is int) {
      inp2Type = i64;
      inp2.ptr.cast<ffi.Int64>().value = scalar;
    } else if (scalar is double) {
      inp2Type = f64;
      inp2.ptr.cast<ffi.Double>().value = scalar;
    } else {
      throw ArgumentError('Scalar must be an int or double');
    }
    final err = op(stream.ptr, out.ptr, inp1.ptr, ffi.nullptr, inp2.ptr, size,
        flip ? 1 : 0, out.type.id, inp1.type.id, inp2Type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void abs(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.abs(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sqr(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.sqr(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sqrt(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.sqrt(
        stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void log(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.log(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void exp(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.exp(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sin(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.sin(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void cos(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.cos(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void tan(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err =
        cuFFI.tan(stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sinh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.sinh(
        stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void cosh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.cosh(
        stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void tanh(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.tanh(
        stream.ptr, out.ptr, inp.ptr, size, out.type.id, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void transpose2D(CudaStream stream, F64Ptr out, F64Ptr inp, Dim3 size) {
    final ctx = Context();
    try {
      final sizePtr = CDim3.from(size);
      final err = cuFFI.transpose2d(stream.ptr, out, inp, sizePtr.ptr.ref);
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
    final err = cuFFI.pickRows(stream.ptr, out.cast(), inp.cast(),
        indices.cast(), sizePtr.ptr.ref, type.id, iType.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void mean(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.mean(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void variance(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    final err = cuFFI.variance(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sum2d(CudaStream stream, NumPtr out, NumPtr inp, Dim2 inpS) {
    final ctx = Context();
    try {
      final sizePtr = CDim2.from(inpS);
      final err = cuFFI.sum2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
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
      final err = cuFFI.mean2d(stream.ptr, out.ptr, inp.ptr, sizePtr.ptr.ref,
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
      final err = cuFFI.variance2d(stream.ptr, out.ptr, inp.ptr,
          sizePtr.ptr.ref, correction, 0, out.type.id, inp.type.id);
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
      final err = cuFFI.variance2d(stream.ptr, out.ptr, inp.ptr,
          sizePtr.ptr.ref, correction, 0xFF, out.type.id, inp.type.id);
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
      final err = cuFFI.normalize2d(stream.ptr, out.ptr, inp.ptr,
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
    final err = cuFFI.matmul(stream.ptr, out, inp1, inp2, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulT(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2, int m,
      int n, int k, int batches) {
    final err = cuFFI.matmulT(stream.ptr, out, inp1, inp2, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulCadd(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2,
      F64Ptr add, int m, int n, int k, int batches) {
    final err =
        cuFFI.matmulCadd(stream.ptr, out, inp1, inp2, add, m, n, k, batches);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void matmulTCadd(CudaStream stream, F64Ptr out, F64Ptr inp1, F64Ptr inp2,
      F64Ptr add, int m, int n, int k, int batches) {
    final err =
        cuFFI.matmulTCadd(stream.ptr, out, inp1, inp2, add, m, n, k, batches);
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

    final err = cuFFI.maxPool2D(
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
    final err = cuFFI.conv2D(
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
    final err = cuFFI.eluActivation(
        stream.ptr, out.ptr, inp.ptr, size, alpha, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void sigmoidActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err = cuFFI.sigmoidActivation(
        stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void siluActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err =
        cuFFI.siluActivation(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void softplusActivation(CudaStream stream, NumPtr out, NumPtr inp, int size,
      int beta, int threshold) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err = cuFFI.softplusActivation(
        stream.ptr, out.ptr, inp.ptr, size, beta, threshold, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }

  void softsignActivation(CudaStream stream, NumPtr out, NumPtr inp, int size) {
    if (inp.type != out.type) {
      throw ArgumentError('Input and output types must be the same');
    }
    final err = cuFFI.softsignActivation(
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
        cuFFI.mishActivation(stream.ptr, out.ptr, inp.ptr, size, inp.type.id);
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
    final err = cuFFI.minThreshold(stream.ptr, out.ptr, inp.ptr,
        thresholdPtr.ptr, valuePtr.ptr, size, inp.type.id);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
  }
}

class CudaStream implements Resource, ffi.Finalizable {
  ffi.Pointer<CCudaStream> cstream;

  CudaStream._(this.cstream, {Context? context}) {
    context?.add(this);
  }

  factory CudaStream(int device, {Context? context}) {
    final stream = CCudaStream.create(device);
    final s = CudaStream._(stream, context: context);
    finalizer.attach(s, stream.cast(), detach: s);
    return s;
  }

  factory CudaStream.noStream(int device, {Context? context}) {
    final stream = ffi.calloc.allocate<CCudaStream>(ffi.sizeOf<CCudaStream>());
    stream.ref.stream = ffi.nullptr;
    stream.ref.deviceId = device;
    final s = CudaStream._(stream, context: context);
    finalizer.attach(s, stream.cast(), detach: s);
    return s;
  }

  ffi.Pointer<CCudaStream> get ptr => cstream;

  int get deviceId => cstream.ref.deviceId;

  Future<void> sync() async {
    if (cstream == ffi.nullptr) return;
    if (cstream.ref.stream == ffi.nullptr) return;

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
      final err = CuFFI.instance!.syncStream(cstream, nc.nativeFunction);
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
    if (cstream == ffi.nullptr) return;

    final err = CuFFI.instance!.destroyStream(cstream);
    if (err != ffi.nullptr) {
      throw CudaException(err.toDartString());
    }
    cstream = ffi.nullptr;
    finalizer.detach(this);
  }

  @override
  void coRelease(Resource other) {
    finalizer.attach(this, other, detach: other);
  }

  @override
  void detachCoRelease(Resource other) {
    finalizer.detach(other);
  }

  static final Finalizer finalizer = Finalizer((other) {
    if (other is ffi.Pointer<CCudaStream>) {
      final err = CuFFI.instance!.destroyStream(other);
      if (err != ffi.nullptr) {
        stdout.writeln(
            'Error destroying stream! CudaException: ${err.toDartString()}');
      }
    } else if (other is Resource) {
      other.release();
    } else {
      stdout.writeln('Invalid type to release: ${other.runtimeType}');
    }
  });
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
    TensorCFFI.finalizer.attach(ret, ptr.cast());
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
    TensorCFFI.finalizer.attach(ret, ret.ptr.cast());
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
  T getOne<T extends num>(
      CudaStream stream, ffi.Pointer<ffi.SizedNativeType> src, NumType<T> type,
      {int index = 0}) {
    final dst = CPtr.allocate(type.bytes);
    memcpy(stream, dst.ptr.cast(),
        src.pointerAddition(index, type.bytes).cast(), type.bytes);
    return type.get(dst.ptr);
  }

  void setOne<T extends num>(CudaStream stream,
      ffi.Pointer<ffi.SizedNativeType> dst, num value, NumType<T> type,
      {int index = 0}) {
    final src = type.allocateForValue(value);
    memcpy(stream, dst.pointerAddition(index, type.bytes).cast(),
        src.ptr.cast(), type.bytes);
  }
}

class _CuPtr<T extends ffi.NativeType> {
  final ffi.Pointer<T> _ptr;

  final int deviceId;

  _CuPtr(this._ptr, this.deviceId);
}

class CuPtr<T extends ffi.NativeType> implements Resource, ffi.Finalizable {
  final _CuPtr<T> _inner;

  CuPtr._(this._inner, {Context? context}) {
    context?.add(this);
    finalizer.attach(this, _inner, detach: this);
  }

  factory CuPtr(ffi.Pointer<T> ptr, int deviceId, {Context? context}) {
    final ret = CuPtr._(_CuPtr(ptr, deviceId), context: context);
    return ret;
  }

  factory CuPtr.allocate(CudaStream stream, int size, {Context? context}) {
    final ptr = cuda.allocate(stream, size);
    return CuPtr<T>(ptr.cast(), stream.deviceId, context: context);
  }

  ffi.Pointer<T> get ptr => _inner._ptr;

  int get deviceId => _inner.deviceId;

  @override
  void release({CudaStream? stream}) {
    // TODO detect if already released
    cuda.memFree(stream ?? CudaStream.noStream(deviceId), _inner._ptr.cast());
    finalizer.detach(this);
  }

  @override
  void coRelease(Resource other) {
    finalizer.attach(this, other, detach: other);
  }

  @override
  void detachCoRelease(Resource other) {
    finalizer.detach(other);
  }

  static Finalizer finalizer = Finalizer((other) {
    try {
      cuda.memFree(CudaStream.noStream(other.deviceId), other._mem.cast());
    } catch (e) {
      stdout.writeln('Error releasing CuPtr: $e');
    }
    other._mem = ffi.nullptr;
  });
}
