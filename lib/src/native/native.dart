export 'c.dart';
export 'cuda.dart';
import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:path/path.dart' as path;

void initializeTensorc() {
  String libraryPath = path.join(Directory.current.path, 'lib', 'asset');
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libtensorc.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorc.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CListFFI.initialize(dylib);
  CudaFFI.initialize(dylib);
}