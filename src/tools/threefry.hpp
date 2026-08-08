#ifndef MRHYDE_THREEFRY_H
#define MRHYDE_THREEFRY_H

#include <cmath>
#include <cstdint>

namespace MrHyDE {

  /**
   * @class Threefry
   * @brief Counter-based random numbers: Threefry-4x64, 20 rounds.
   *        Salmon, Moraes, Dror & Shaw, "Parallel Random Numbers: As Easy as
   *        1, 2, 3", SC'11.
   *
   * Every draw is a pure function of (seed, stream, substream, index), so a
   * sketch can regenerate its test-matrix rows bit-identically instead of
   * storing them -- the point of the seed-rematerialized design.  There is no
   * internal state and no sequencing requirement; any element can be produced
   * at any time, in any order, on any rank.
   *
   * The raw counter words are portable across platforms.  normal() maps them
   * through Box-Muller (log/sqrt/cos), so those doubles are deterministic
   * within a build but inherit the platform's libm across builds.
   */
  class Threefry {

  public:

    Threefry(const uint64_t & seed, const uint64_t & stream) {
      key_[0] = seed;
      key_[1] = stream;
      key_[2] = 0;
      key_[3] = 0;
    }

    /**
     * @brief The Threefry-4x64-20 block cipher: 4 counter words in, 4 words out.
     */
    static void encrypt(const uint64_t counter[4], const uint64_t key[4], uint64_t out[4]) {

      // rotation schedule for the 4x64 variant, cycle of 8 double-rounds
      static const int rot[8][2] = {
        {14, 16}, {52, 57}, {23, 40}, { 5, 37},
        {25, 33}, {46, 12}, {58, 22}, {32, 32}
      };
      const uint64_t parity = 0x1BD11BDAA9FC1A22ULL;

      uint64_t ks[5];
      ks[0] = key[0];  ks[1] = key[1];  ks[2] = key[2];  ks[3] = key[3];
      ks[4] = parity ^ ks[0] ^ ks[1] ^ ks[2] ^ ks[3];

      uint64_t x0 = counter[0] + ks[0];
      uint64_t x1 = counter[1] + ks[1];
      uint64_t x2 = counter[2] + ks[2];
      uint64_t x3 = counter[3] + ks[3];

      for (int round=0; round<20; ++round) {

        x0 += x1;  x1 = rotl(x1, rot[round%8][0]);  x1 ^= x0;
        x2 += x3;  x3 = rotl(x3, rot[round%8][1]);  x3 ^= x2;

        // 4x64 word permutation: (x0,x1,x2,x3) -> (x0,x3,x2,x1)
        const uint64_t tmp = x1;
        x1 = x3;
        x3 = tmp;

        if ((round+1)%4 == 0) {
          const uint64_t inject = static_cast<uint64_t>((round+1)/4);
          x0 += ks[(inject + 0)%5];
          x1 += ks[(inject + 1)%5];
          x2 += ks[(inject + 2)%5];
          x3 += ks[(inject + 3)%5];
          x3 += inject;
        }
      }

      out[0] = x0;  out[1] = x1;  out[2] = x2;  out[3] = x3;
    }

    /**
     * @brief Uniform draw in (0,1), element i of the given substream.
     */
    double uniform(const uint64_t & substream, const uint64_t & i) const {
      uint64_t words[4];
      block(substream, i/4, words);
      return toUnit(words[i%4]);
    }

    /**
     * @brief Standard normal draw, element i of the given substream (Box-Muller).
     */
    double normal(const uint64_t & substream, const uint64_t & i) const {
      double z[4];
      normalBlock(substream, i/4, z);
      return z[i%4];
    }

    /**
     * @brief Fill out[0..count) with normals: elements first..first+count-1 of
     *        the substream.  Identical values to calling normal() per element.
     */
    void fillNormal(const uint64_t & substream, const uint64_t & first,
                    const int & count, double * out) const {
      double z[4];
      uint64_t cached_block = ~uint64_t(0);
      for (int n=0; n<count; ++n) {
        const uint64_t i = first + static_cast<uint64_t>(n);
        if (i/4 != cached_block) {
          cached_block = i/4;
          normalBlock(substream, cached_block, z);
        }
        out[n] = z[i%4];
      }
    }

  private:

    static uint64_t rotl(const uint64_t & x, const int & n) {
      return (x << n) | (x >> (64 - n));
    }

    /// Map a counter word to (0,1), endpoints excluded so log() is safe.
    static double toUnit(const uint64_t & x) {
      return (static_cast<double>(x >> 11) + 0.5)*(1.0/9007199254740992.0);   // 2^-53
    }

    /// Raw words for block b of a substream.
    void block(const uint64_t & substream, const uint64_t & b, uint64_t words[4]) const {
      const uint64_t counter[4] = {b, substream, 0, 0};
      encrypt(counter, key_, words);
    }

    /// Four normals for block b of a substream: two Box-Muller pairs.
    void normalBlock(const uint64_t & substream, const uint64_t & b, double z[4]) const {
      uint64_t words[4];
      block(substream, b, words);
      const double two_pi = 6.283185307179586476925286766559;
      const double u0 = toUnit(words[0]);
      const double u1 = toUnit(words[1]);
      const double u2 = toUnit(words[2]);
      const double u3 = toUnit(words[3]);
      const double r0 = std::sqrt(-2.0*std::log(u0));
      const double r1 = std::sqrt(-2.0*std::log(u2));
      z[0] = r0*std::cos(two_pi*u1);
      z[1] = r0*std::sin(two_pi*u1);
      z[2] = r1*std::cos(two_pi*u3);
      z[3] = r1*std::sin(two_pi*u3);
    }

    uint64_t key_[4];   ///< (seed, stream, 0, 0)
  };

}

#endif
