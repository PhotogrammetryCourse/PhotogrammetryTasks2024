#include "pm_fast_random.h"

#include <libutils/rasserts.h>


// Returns pseudo-random value in range [min; max] (inclusive)
unsigned int phg::FastRandom::next(unsigned int min, unsigned int max) {
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    unsigned int t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    unsigned int result;
    if (min == 0 && max == std::numeric_limits<unsigned int>::max()) {
        result = z;
    } else {
        result = min + (z % (max - min + 1));
    }

    rassert(result >= min, 2391012512037);
    rassert(result <= max, 2391012512038);
    return result;
}

float phg::FastRandom::nextf(float min, float max) {
    unsigned int big_prime = 1000 * 1000 * 1000 + 7;
    float result = next(0, big_prime) * 1.0f / big_prime;

    rassert(result >= 0.0f, 23910121261046);
    rassert(result <= 1.0f, 23910121261047);
    result = min + result * (max - min);
    rassert(result >= min, 23910121261049);
    rassert(result <= max, 23910121261050);
    return result;
}

vector3f phg::FastRandom::nextPointFromSphere() {
    float q1, q2, S;
    do {
        q1 = nextf(-1.0f, 1.0f);
        q2 = nextf(-1.0f, 1.0f);
        S = q1 * q1 + q2 * q2;
    } while (S >= 1.0f);
    float nx = 1 - 2 * S;
    float sqrt1minS = sqrtf(1.0f - S);
    float ny = 2 * q1 * sqrt1minS;
    float nz = 2 * q2 * sqrt1minS;
    rassert(nx * nx + ny * ny + nz * nz >= 0.99f, 2391012235161069);
    rassert(nx * nx + ny * ny + nz * nz <= 1.01f, 2391012235161069);
    return vector3f(nx, ny, nz);
}
