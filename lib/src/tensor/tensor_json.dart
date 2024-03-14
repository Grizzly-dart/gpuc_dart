import 'dart:convert';

import 'package:gpuc_dart/gpuc_dart.dart';

enum TensorJsonType {
  IntData,
  DoubleData,
  TensorData,
  SizeData,
  StringData,
}

abstract class TensorJsonData {
  String get name;

  static TensorJsonData fromMap(Map map) {
    final type = TensorJsonType.values.byName(map['type']);
    if(type == TensorJsonType.TensorData) {
      return TensorJsonTensor.fromMap(map);
    }
    throw UnsupportedError('$type not supported');
  }

  static Map<String, TensorJsonData> fromList(List list) {
    final ret = <String, TensorJsonData>{};
    for (final item in list) {
      final t = TensorJsonData.fromMap(item);
      ret[t.name] = t;
    }
    return ret;
  }
}

class TensorJsonTensor implements TensorJsonData {
  final String name;
  final Dim size;
  final List<double> data;

  TensorJsonType get type => TensorJsonType.TensorData;

  TensorJsonTensor(
      {required this.name, required this.size, required this.data});

  factory TensorJsonTensor.fromMap(Map map) =>
      TensorJsonTensor(
          name: map['name'],
          size: Dim.from(map['size']),
          data: (map['data'] as List).cast());

  Map<String, dynamic> toJson() => {
        'name': name,
        'size': size.toList(),
        'type': type.name,
        'data': data.toList(),
      };
}

class TensorJsonArgs {}

class TensorJsonFile {}

Map<String, TensorJson> parseTensorJson(String str) {
  final list = jsonDecode(str);
  return TensorJson.fromMapList(list);
}
