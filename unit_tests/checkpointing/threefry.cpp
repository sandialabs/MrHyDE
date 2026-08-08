#include "threefry.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

using namespace MrHyDE;

int main(){

    int num_failures = 0;

    // --- known-answer vectors for Threefry-4x64-20, verbatim from the
    // Random123 reference distribution (tests/kat_vectors).  If these pass,
    // the cipher is correct.
    {
        const uint64_t zero[4] = {0, 0, 0, 0};
        const uint64_t want_zero[4] = {0x09218ebde6c85537ULL, 0x55941f5266d86105ULL,
                                       0x4bd25e16282434dcULL, 0xee29ec846bd2e40bULL};

        const uint64_t pi_ctr[4] = {0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
                                    0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL};
        const uint64_t pi_key[4] = {0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
                                    0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL};
        const uint64_t want_pi[4] = {0xa7e8fde591651bd9ULL, 0xbaafd0c30138319bULL,
                                     0x84a5c1a729e685b9ULL, 0x901d406ccebc1ba4ULL};

        uint64_t got[4];

        Threefry::encrypt(zero, zero, got);
        for (int i=0; i<4; ++i) {
            if (got[i] != want_zero[i]) {
                std::cout << "FAIL: zero-vector KAT word " << i << std::endl;
                ++num_failures;
            }
        }

        Threefry::encrypt(pi_ctr, pi_key, got);
        for (int i=0; i<4; ++i) {
            if (got[i] != want_pi[i]) {
                std::cout << "FAIL: pi-vector KAT word " << i << std::endl;
                ++num_failures;
            }
        }
    }

    // --- determinism and random access: any element, any order, same value
    {
        Threefry rng(42, 3);
        const double a = rng.normal(7, 12345);
        const double b = rng.normal(7, 12345);
        if (a != b) {
            std::cout << "FAIL: repeated draw differs" << std::endl;
            ++num_failures;
        }

        double filled[40];
        rng.fillNormal(7, 12320, 40, filled);
        bool match = true;
        for (int n=0; n<40; ++n) {
            if (filled[n] != rng.normal(7, 12320 + n)) { match = false; }
        }
        if (!match) {
            std::cout << "FAIL: fillNormal disagrees with per-element normal" << std::endl;
            ++num_failures;
        }
    }

    // --- distinct (seed, stream, substream) must give distinct sequences
    {
        Threefry rng_a(42, 3), rng_b(42, 4), rng_c(43, 3);
        int same_ab = 0, same_ac = 0, same_sub = 0;
        for (int i=0; i<100; ++i) {
            if (rng_a.normal(0, i) == rng_b.normal(0, i)) { ++same_ab; }
            if (rng_a.normal(0, i) == rng_c.normal(0, i)) { ++same_ac; }
            if (rng_a.normal(0, i) == rng_a.normal(1, i)) { ++same_sub; }
        }
        if (same_ab > 0 || same_ac > 0 || same_sub > 0) {
            std::cout << "FAIL: streams/substreams are not distinct" << std::endl;
            ++num_failures;
        }
    }

    // --- moments: with n = 4e6 samples the standard error of the mean is
    // 5e-4, of the variance about 7e-4; allow five standard errors
    {
        Threefry rng(2026, 0);
        const long n = 4000000;
        double sum = 0.0, sum_sq = 0.0;
        double buffer[1000];
        for (long start=0; start<n; start+=1000) {
            rng.fillNormal(0, start, 1000, buffer);
            for (int i=0; i<1000; ++i) {
                sum += buffer[i];
                sum_sq += buffer[i]*buffer[i];
            }
        }
        const double mean = sum/n;
        const double var = sum_sq/n - mean*mean;
        if (std::abs(mean) > 2.5e-3) {
            std::cout << "FAIL: normal mean " << mean << " too far from 0" << std::endl;
            ++num_failures;
        }
        if (std::abs(var - 1.0) > 3.5e-3) {
            std::cout << "FAIL: normal variance " << var << " too far from 1" << std::endl;
            ++num_failures;
        }
    }

    // --- uniforms stay strictly inside (0,1) so log() in Box-Muller is safe
    {
        Threefry rng(7, 7);
        bool inside = true;
        for (int i=0; i<100000; ++i) {
            const double u = rng.uniform(0, i);
            if (u <= 0.0 || u >= 1.0) { inside = false; }
        }
        if (!inside) {
            std::cout << "FAIL: uniform left (0,1)" << std::endl;
            ++num_failures;
        }
    }

    if (num_failures == 0) {
        std::cout << "All Threefry tests PASSED" << std::endl;
    }

    return num_failures == 0 ? 0 : 1;
}
