#pragma once
#include <random>
#include <limits>
// The random seed used in debug and release mode
extern const size_t rand_seed;
extern std::mt19937_64 engine;

#define INF std::numeric_limits<double>::infinity()
