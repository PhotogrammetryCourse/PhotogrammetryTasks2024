#pragma once

#include <limits>

#include <phg/sfm/defines.h>


namespace phg {

    // See https://stackoverflow.com/a/1640399
    class FastRandom {
    public:
        FastRandom(unsigned int globalSeed, unsigned int localSeed) {
            unsigned int big_prime = 1000 * 1000 * 1000 + 7;
            reset(globalSeed * big_prime + localSeed * 239017);
        }

        FastRandom(unsigned int seed = 123456789) {
            reset(seed);
        }

        void reset(unsigned int seed = 123456789) {
            x = seed;
            y = 362436069;
            z = 521288629;
        }

        // Returns pseudo-random value in range [min; max] (inclusive)
        unsigned int next(unsigned int min = 0, unsigned int max = std::numeric_limits<unsigned int>::max());

        // Returns pseudo-random value in range [0; 1] (inclusive)
        float nextf(float min = 0.0f, float max = 1.0f);

        vector3f nextPointFromSphere();

    private:
        unsigned int x, y, z;
    };

}
