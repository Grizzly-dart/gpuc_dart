import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

class CListView extends NList with CListMixin, ListMixin<double> implements CList {
  final CList _list;

  final int _offset;

  final int _length;

  CListView(this._list, this._offset, this._length);

  @override
  DeviceType get deviceType => _list.deviceType;

  @override
  int get deviceId => _list.deviceId;

  @override
  int get length => _length;

  @override
  int get lengthBytes => _length * NList.byteSize;

  @override
  double operator [](int index) => _list[_offset + index];

  @override
  void operator []=(int index, double value) => _list[_offset + index] = value;

  @override
  ffi.Pointer<ffi.Double> get ptr => _list.ptr + _offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
