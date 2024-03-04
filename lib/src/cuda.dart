import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

import 'package:path/path.dart' as path;
import 'cpu.dart';

void init() {
  String libraryPath = path.join(Directory.current.path, 'lib', 'asset');
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libgpuc_cuda.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libgpuc_cuda.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libgpuc_cuda.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  _CudaListMethods.make1D = dylib.lookupFunction<
      _CudaTensor Function(ffi.Uint64),
      _CudaTensor Function(int)>('makeTensor1D');
  _CudaListMethods.release = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor),
      void Function(_CudaTensor)>('makeTensor1D');

  _CudaListMethods.writeTensor = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor, ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(_CudaTensor, ffi.Pointer<ffi.Double>, int)>('writeTensor');
  _CudaListMethods.readTensor = dylib.lookupFunction<
      ffi.Void Function(_CudaTensor, ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(_CudaTensor, ffi.Pointer<ffi.Double>, int)>('readTensor');

  elementwiseAdd2 = dylib.lookupFunction<
      ffi.Void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, ffi.Uint64),
      void Function(ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>,
          ffi.Pointer<ffi.Double>, int)>('elementwiseAdd2');
}
