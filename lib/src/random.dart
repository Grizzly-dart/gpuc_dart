import 'dart:math';
import 'package:mt19937/mt19937.dart';

/// Makes testing across C, Python and Dart easier when all of them generate the
/// same random numbers
class MTRandom implements Random {
  final MersenneTwister mt;

  MTRandom({int seed = 0}) : mt = MersenneTwister(seed: seed);

  @override
  bool nextBool() => mt.genRandInt32() > 0x7FFFFFFF;

  @override
  double nextDouble() => mt.genRandInt32() / 0xFFFFFFFF;

  @override
  int nextInt(int max) => mt.genRandInt32() % max;
}
