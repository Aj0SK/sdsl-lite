#ifndef INCLUDED_SDSL_RRR_VECTOR_SPEC_HELPERS
#define INCLUDED_SDSL_RRR_VECTOR_SPEC_HELPERS

#include "int_vector.hpp"
#include "util.hpp"
#include "rrr_helper.hpp" // for binomial helper class
#include "iterators.hpp"
#include <vector>
#include <algorithm> // for next_permutation
#include <iostream>

#ifdef __SSE4_2__
#include <xmmintrin.h>
#endif

namespace sdsl
{

// Helper class for the binomial coefficients \f$ 15 \choose k \f$
/*
 * Size of lookup tables:
 *  * m_nr_to_bin: 64 kB = (2^15 entries x 2 bytes)
 *  * m_bin_to_nr: 64 kB = (2^15 entries x 2 bytes)
 */
class binomial15
{
    public:
        typedef uint32_t number_type;
    private:

        static class impl
        {
            public:
                static const int n = 15;
                static const int MAX_SIZE=32;
                uint8_t m_space_for_bt[16];
                uint8_t m_space_for_bt_pair[256];
                uint64_t m_C[MAX_SIZE];
                int_vector<16> m_nr_to_bin;
                int_vector<16> m_bin_to_nr;

                impl()
                {
                    m_nr_to_bin.resize(1<<n);
                    m_bin_to_nr.resize(1<<n);
                    for (int i=0, cnt=0, class_cnt=0; i<=n; ++i) {
                        m_C[i] = cnt;
                        class_cnt = 0;
                        std::vector<bool> b(n,0);
                        for (int j=0; j<i; ++j) b[n-j-1] = 1;
                        do {
                            uint32_t x=0;
                            for (int k=0; k<n; ++k)
                                x |= ((uint32_t)b[n-k-1])<<(n-1-k);
                            m_nr_to_bin[cnt] = x;
                            m_bin_to_nr[x] = class_cnt;
                            ++cnt;
                            ++class_cnt;
                        } while (next_permutation(b.begin(), b.end()));
                        if (class_cnt == 1)
                            m_space_for_bt[i] = 0;
                        else
                            m_space_for_bt[i] = bits::hi(class_cnt)+1;
                    }
                    if (n == 15) {
                        for (int x=0; x<256; ++x) {
                            m_space_for_bt_pair[x] = m_space_for_bt[x>>4] + m_space_for_bt[x&0x0F];
                        }
                    }
                }
        } iii;

    public:

        static inline uint8_t space_for_bt(uint32_t i)
        {
            return iii.m_space_for_bt[i];
        }

        static inline uint32_t nr_to_bin(uint8_t k, uint32_t nr)
        {
            return iii.m_nr_to_bin[iii.m_C[k]+nr];
        }

        static inline uint32_t bin_to_nr(uint32_t bin)
        {
            return iii.m_bin_to_nr[bin];
        }

        static inline uint8_t space_for_bt_pair(uint8_t x)
        {
            return iii.m_space_for_bt_pair[x];
        }
};

class binomial31
{
    public:
        typedef uint32_t number_type;
    private:

        static class impl
        {
            public:
                std::array<uint64_t, 64> m_bin_15 = {0};
                std::array<uint64_t, 64> m_bin_30 = {0};
                // max number stored is roughly 16 * 41'409'225
                std::array<std::array<uint32_t, 16>, 31> helper;
                uint8_t m_space_for_bt[32];

                impl()
                {
                    binomial_table<31, uint64_t> m_bin_table;
                    for (int i = 0; i <= 15; ++i)
                        m_bin_15[i] = m_bin_table.data.table[15][i];
                    for (int i = 0; i <= 30; ++i)
                        m_bin_30[i] = m_bin_table.data.table[30][i];
                    for (int i = 0; i < 32; ++i)
                    {
                        size_t class_cnt = m_bin_table.data.table[31][i];
                        if (class_cnt == 1)
                            m_space_for_bt[i] = 0;
                        else
                            m_space_for_bt[i] = bits::hi(class_cnt) + 1;
                    }

                    for (size_t k = 0; k < 31; ++k)
                    {
                        std::fill(helper[k].begin(), helper[k].end(), 0);
                        uint32_t total = 0;
                        for (size_t ones_in_big = (k > 15) ? (k - 15) : 0;
                            ones_in_big <= std::min(k, static_cast<size_t>(15)); ++ones_in_big)
                        {
                            helper[k][ones_in_big] = total;
                            total += m_bin_15[ones_in_big] * m_bin_15[k - ones_in_big];
                        }
                    }
                } // impl() end
        } iii;

    public:
        static inline uint32_t nr_to_bin_30(const uint8_t k, uint32_t nr)
        {
            const int right_k_from = (k > 15) ? (k - 15) : 0;
            const int right_k_to = std::min(k, static_cast<uint8_t>(15));

        #ifdef __SSE4_2__
            const __m128i keys = _mm_set1_epi32(nr);
            const __m128i vec1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&iii.helper[k][0]));
            const __m128i vec2 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&iii.helper[k][4]));
            const __m128i vec3 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&iii.helper[k][8]));
            const __m128i vec4 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&iii.helper[k][12]));

            const __m128i cmp1 = _mm_cmpgt_epi32(vec1, keys);
            const __m128i cmp2 = _mm_cmpgt_epi32(vec2, keys);
            const __m128i cmp3 = _mm_cmpgt_epi32(vec3, keys);
            const __m128i cmp4 = _mm_cmpgt_epi32(vec4, keys);

            const __m128i tmp1 = _mm_packs_epi32(cmp1, cmp2);
            const __m128i tmp2 = _mm_packs_epi32(cmp3, cmp4);
            const uint32_t mask1 = _mm_movemask_epi8(tmp1);
            const uint32_t mask2 = _mm_movemask_epi8(tmp2);

            const uint32_t mask = (mask2 << 16) | mask1;

            int right_k = right_k_to;

            if (mask != 0)
            {
                right_k = (1 + __builtin_ctz(mask)) / 2;

                if (iii.helper[k][right_k] > nr)
                    --right_k;
            }
        #else
            int right_k = right_k_from;
            for (; right_k < right_k_to; ++right_k)
            {
                if (auto curr_index = iii.helper[k][right_k + 1]; curr_index >= nr)
                {
                    if (curr_index == nr)
                    ++right_k;
                    break;
                }
            }
        #endif

            nr -= iii.helper[k][right_k];

            int left_k = k - right_k;

            uint32_t left_bin =
                binomial15::nr_to_bin(left_k, nr % iii.m_bin_15[left_k]);

            uint32_t right_bin =
                binomial15::nr_to_bin(right_k, nr / iii.m_bin_15[left_k]);

            return (left_bin << 15) | right_bin;
        }

        static inline uint32_t bin_to_nr_30(const uint32_t bin)
        {
            int k = __builtin_popcount(bin);
            uint32_t nr = 0;
            uint32_t left_bin = bin >> 15;
            uint32_t right_bin = bin & 0x7fff; // lower 15 bits
            int left_k = __builtin_popcount(left_bin);
            int right_k = __builtin_popcount(right_bin);

            nr += iii.helper[k][right_k];
            nr += iii.m_bin_15[left_k] * binomial15::bin_to_nr(right_bin);
            nr += binomial15::bin_to_nr(left_bin);

            return nr;
        }

        static inline uint32_t nr_to_bin(uint8_t k, uint32_t nr)
        {
        #ifndef NO_MY_OPT
            if (k == 31)
            {
                return (1ull << 31) - 1;
            }
            else if (k == 0)
            {
                return 0;
            }
            else if (k == 1)
            {
                return (nr >= 30) ? (1ull << nr) : (1ull << (30 - nr - 1));
            }
        #endif
            const uint32_t threshold = iii.m_bin_30[k];
            uint32_t to_or = 0;
            if (nr >= threshold)
            {
                --k;
                nr -= threshold;
                to_or = 1 << 30;
            }

            uint32_t bin = to_or | nr_to_bin_30(k, nr);
            return bin;
        }

        static inline uint32_t bin_to_nr(const uint32_t bin)
        {
            uint32_t k = __builtin_popcount(bin);

        #ifndef NO_MY_OPT
            if ((bin == 0) || (bin == ((1ull << 31) - 1)))
            {
                return 0;
            }
        #endif

            uint32_t nr;
            if (bin & (1 << 30))
            {
                uint32_t new_bin = bin & (~(1 << 30));
                nr = iii.m_bin_30[k] + bin_to_nr_30(new_bin);
            }
            else
            {
                nr = bin_to_nr_30(bin);
            }
            return nr;
        }

        static inline uint8_t space_for_bt(uint32_t i)
        {
            return iii.m_space_for_bt[i];
        }

        //! Decode the bit at position \f$ off \f$ of the block encoded by the pair
        //! (k, nr).
        static inline bool decode_bit(uint16_t k, number_type nr, uint16_t off)
        {
            return (nr_to_bin(k, nr) >> off) & (uint32_t)1;
        }
};

}// end namespace sdsl

#endif