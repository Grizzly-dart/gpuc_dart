import 'dart:convert';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn2d/nn2d.dart';

export 'tenson_cmd.dart';

enum TensonType<T> {
  intData<int>('Int'),
  doubleData<double>('Double'),
  stringData<String>('String'),
  dimData<Dim>('Dim'),
  tensorData<F64Tensor>('Tensor'),
  nullData<Null>('Null'),
  ;

  final String jsonName;

  const TensonType(this.jsonName);

  String toJson() => jsonName;

  bool typeMatch(v) {
    if (v == null) return this == TensonType.nullData;
    return v is T;
  }

  static TensonType fromJson(String jsonName) =>
      values.firstWhere((e) => e.jsonName == jsonName);
}

class TensonVar<T> {
  final String name;
  final T data;

  late final TensonType dataType;

  TensonVar({required this.name, required this.data}) {
    if (data != null) {
      dataType = TensonType.values.firstWhere((e) => e.typeMatch(data));
    } else {
      dataType = TensonType.nullData;
    }
  }

  static TensonVar fromMap(Map map) {
    final type = TensonType.fromJson(map['type']);
    if (type == TensonType.intData) {
      return TensonVar(name: map['name'], data: map['data']);
    } else if (type == TensonType.doubleData) {
      return TensonVar<double>(name: map['name'], data: map['data']);
    } else if (type == TensonType.stringData) {
      return TensonVar<String>(name: map['name'], data: map['data']);
    } else if (type == TensonType.dimData) {
      return TensonVar<Dim>(name: map['name'], data: Dim.from(map['data']));
    } else if (type == TensonType.tensorData) {
      final dim = Dim.from(map['data']['size']);
      final data = (map['data']['data'] as List).cast<double>();
      return TensonVar<F64Tensor>(
          name: map['name'],
          data: F64Tensor.fromList(data, size: dim, name: map['name']));
    }
    throw UnsupportedError('$type not supported');
  }

  static Map<String, TensonVar> mapFromList(List list) {
    final map = <String, TensonVar>{};
    for (final item in list) {
      final v = TensonVar.fromMap(item);
      map[v.name] = v;
    }
    return map;
  }

  Map<String, dynamic> toJson() => {
        'name': name,
        'type': dataType,
        'data': data,
      };
}

Map<String, TensonVar> parseTenson(String str) {
  final list = jsonDecode(str);
  return TensonVar.mapFromList(list);
}
