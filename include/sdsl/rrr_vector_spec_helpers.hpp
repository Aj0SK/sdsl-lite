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

template<bool is_hybrid, uint16_t cutoff=31>
class binomial31
{
    public:
        typedef uint32_t number_type;
    private:
        std::array<uint64_t, 64> m_bin_15 = {0};
        std::array<uint64_t, 64> m_bin_30 = {0};
        // max number stored is roughly 16 * 41'409'225
        std::array<std::array<uint32_t, 16>, 31> helper;
        uint8_t m_space_for_bt[32];
    public:
        binomial31()
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
                else if (is_hybrid && i >= cutoff)
                    m_space_for_bt[i] = 31;
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
        } // binomial31 constructors end

        inline uint32_t nr_to_bin_30(const uint8_t k, uint32_t nr) const
        {
            const int right_k_from = (k > 15) ? (k - 15) : 0;
            const int right_k_to = std::min(k, static_cast<uint8_t>(15));

        #ifdef __SSE4_2__
            const __m128i keys = _mm_set1_epi32(nr);
            const __m128i vec1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&helper[k][0]));
            const __m128i vec2 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&helper[k][4]));
            const __m128i vec3 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&helper[k][8]));
            const __m128i vec4 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&helper[k][12]));

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

                if (helper[k][right_k] > nr)
                    --right_k;
            }
        #else
            int right_k = right_k_from;
            for (; right_k < right_k_to; ++right_k)
            {
                if (auto curr_index = helper[k][right_k + 1]; curr_index >= nr)
                {
                    if (curr_index == nr)
                    ++right_k;
                    break;
                }
            }
        #endif

            nr -= helper[k][right_k];

            int left_k = k - right_k;

            uint32_t left_bin =
                binomial15::nr_to_bin(left_k, nr % m_bin_15[left_k]);

            uint32_t right_bin =
                binomial15::nr_to_bin(right_k, nr / m_bin_15[left_k]);

            return (left_bin << 15) | right_bin;
        }

        inline uint32_t bin_to_nr_30(const uint32_t bin) const
        {
            int k = __builtin_popcount(bin);
            uint32_t nr = 0;
            uint32_t left_bin = bin >> 15;
            uint32_t right_bin = bin & 0x7fff; // lower 15 bits
            int left_k = __builtin_popcount(left_bin);
            int right_k = __builtin_popcount(right_bin);

            nr += helper[k][right_k];
            nr += m_bin_15[left_k] * binomial15::bin_to_nr(right_bin);
            nr += binomial15::bin_to_nr(left_bin);

            return nr;
        }

        inline uint32_t nr_to_bin(uint8_t k, uint32_t nr) const
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

            if (is_hybrid && k >= cutoff)
            {
                return nr;
            }

            const uint32_t threshold = m_bin_30[k];
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

        inline uint32_t bin_to_nr(const uint32_t bin) const
        {
            uint32_t k = __builtin_popcount(bin);

        #ifndef NO_MY_OPT
            if ((bin == 0) || (bin == ((1ull << 31) - 1)))
            {
                return 0;
            }
        #endif

            if (is_hybrid && k >= cutoff)
            {
                return bin;
            }

            uint32_t nr;
            if (bin & (1 << 30))
            {
                uint32_t new_bin = bin & (~(1 << 30));
                nr = m_bin_30[k] + bin_to_nr_30(new_bin);
            }
            else
            {
                nr = bin_to_nr_30(bin);
            }
            return nr;
        }

        inline uint8_t space_for_bt(uint32_t i) const
        {
            return m_space_for_bt[i];
        }

        //! Decode the bit at position \f$ off \f$ of the block encoded by the pair
        //! (k, nr).
        inline bool decode_bit(uint16_t k, number_type nr, uint16_t off) const
        {
            return (nr_to_bin(k, nr) >> off) & (uint32_t)1;
        }
};

template<bool is_hybrid, uint16_t cutoff=31>
class binomial63
{
    public:
        typedef uint64_t number_type;
    private:
        std::array<uint64_t, 64> m_bin_30 = {0};
        std::array<uint64_t, 64> m_bin_60 = {0};
        std::array<uint64_t, 64> m_bin_61 = {0};
        std::array<uint64_t, 64> m_bin_62 = {0};
        std::array<std::array<uint64_t, 31>, 64> helper;
        uint32_t m_space_for_bt[64];
        binomial31<false> helper31;
    public:
        binomial63()
        {
            binomial_table<63, uint64_t> m_bin_table;
            for (uint64_t i = 0; i <= 30; ++i)
                m_bin_30[i] = m_bin_table.data.table[30][i];

            for (uint64_t i = 0; i <= 60; ++i)
                m_bin_60[i] = m_bin_table.data.table[60][i];

            for (uint64_t i = 0; i <= 61; ++i)
                m_bin_61[i] = m_bin_table.data.table[61][i];

            for (uint64_t i = 0; i <= 62; ++i)
                m_bin_62[i] = m_bin_table.data.table[62][i];

            for (uint64_t i = 0; i < 64; ++i)
            {
                uint64_t class_cnt = m_bin_table.data.table[63][i];
                if (class_cnt == 1)
                    m_space_for_bt[i] = 0;
                else if (is_hybrid && i >= cutoff)
                    m_space_for_bt[i] = 63;
                else
                    m_space_for_bt[i] = bits::hi(class_cnt) + 1;
            }

            for (size_t k = 0; k < 64; ++k)
            {
                std::fill(helper[k].begin(), helper[k].end(), 0);
                uint64_t total = 0;
                for (size_t ones_in_big = (k > 30) ? (k - 30) : 0;
                    ones_in_big <= std::min(k, static_cast<size_t>(30)); ++ones_in_big)
                {
                    helper[k][ones_in_big] = total;
                    total += m_bin_30[ones_in_big] * m_bin_30[k - ones_in_big];
                }
            }
        } // binomial63 constructor end

        public:
            inline uint64_t nr_to_bin(uint8_t k, uint64_t nr) const
            {
            #ifndef NO_MY_OPT
                if (k == 63)
                {
                    return (1ull << 63) - 1;
                }
                else if (k == 0)
                {
                    return 0;
                }
                else if (k == 1)
                {
                    return (nr >= 60) ? (1ull << nr) : (1ull << (60 - nr - 1));
                }
            #endif
                if (is_hybrid && k >= cutoff)
                {
                    return nr;
                }

                uint64_t to_or = 0;
                const bool first_bit = (nr >= m_bin_62[k]);
                if (first_bit)
                {
                    nr -= m_bin_62[k];
                    --k;
                    to_or |= 1ull << 62;
                }
                const bool second_bit = (nr >= (m_bin_61[k]));
                if (second_bit)
                {
                    nr -= m_bin_61[k];
                    --k;
                    to_or |= 1ull << 61;
                }
                const bool third_bit = (nr >= (m_bin_60[k]));
                if (third_bit)
                {
                    nr -= m_bin_60[k];
                    --k;
                    to_or |= 1ull << 60;
                }

                const size_t right_k_from = (k > 30) ? (k - 30) : 0;
                const size_t right_k_to = std::min(k, static_cast<uint8_t>(30));

                size_t right_k = right_k_from;
                for (; right_k < right_k_to; ++right_k)
                {
                    if (auto curr_index = helper[k][right_k + 1]; curr_index >= nr)
                    {
                        if (curr_index == nr)
                        ++right_k;
                        break;
                    }
                }

                nr -= helper[k][right_k];

                const size_t left_k = k - right_k;

                const uint64_t left_bin =
                    helper31.nr_to_bin_30(left_k, nr % m_bin_30[left_k]);
                const uint64_t right_bin =
                    helper31.nr_to_bin_30(right_k, nr / m_bin_30[left_k]);

                return to_or | (left_bin << 30) | right_bin;
            }

            inline uint64_t bin_to_nr(const uint64_t bin) const
            {
                const uint64_t k = __builtin_popcountll(bin);

                if (is_hybrid && k >= cutoff)
                {
                    return bin;
                }

            #ifndef NO_MY_OPT
                if ((bin == 0) || (bin == ((1ull << 63) - 1)))
                {
                    return 0;
                }
            #endif

                bool first_bit = bin & (1ull << 62);
                bool second_bit = bin & (1ull << 61);
                bool third_bit = bin & (1ull << 60);

                uint64_t nr = first_bit * m_bin_62[k] +
                                second_bit * m_bin_61[k - first_bit] +
                                third_bit * m_bin_60[k - first_bit - second_bit];

                const uint64_t rem_k = k - first_bit - second_bit - third_bit;

                // 1073741823_10 = 111111111111111111111111111111_2
                const uint32_t left_bin = (bin >> 30) & 1073741823ull;
                const uint32_t right_bin = bin & 1073741823ull;
                const uint32_t left_k = __builtin_popcount(left_bin);
                const uint32_t right_k = __builtin_popcount(right_bin);

                nr += helper[rem_k][right_k];
                nr += m_bin_30[left_k] * helper31.bin_to_nr_30(right_bin);
                nr += helper31.bin_to_nr_30(left_bin);

                return nr;
            }

            inline uint32_t space_for_bt(uint32_t i) const
            {
                return m_space_for_bt[i];
            }

            //! Decode the bit at position \f$ off \f$ of the block encoded by the pair
            //! (k, nr).
            inline bool decode_bit(uint16_t k, number_type nr, uint16_t off) const
            {
            #ifndef NO_MY_OPT
                if (k == 63)
                {
                    return 1;
                }
                else if (k == 0)
                {
                    return 0;
                }
                else if (k == 1)
                {
                    return (nr >= 60) ? (nr == off) : ((60 - nr - 1) == off);
                }
            #endif
                return (nr_to_bin(k, nr) >> off) & static_cast<uint64_t>(1);
            }
};

class binomial127
{
    public:
        typedef __uint128_t number_type;
    private:
        static class impl
        {
            public:
                std::array<__uint128_t, 128> m_bin_63 = {0};
                std::array<__uint128_t, 128> m_bin_126 = {0};
                std::array<__uint128_t, 128> m_bin_127 = {0};
                std::array<std::array<__uint128_t, 64>, 128> helper;
                uint32_t m_space_for_bt[128];
                binomial63<false> helper63;

                impl()
                {
                    binomial_table<127, __uint128_t> m_bin_table;

                    for(int i=0; i<128; ++i)
                        m_bin_63[i] = m_bin_table.data.table[63][i];

                    for(int i=0; i<128; ++i)
                        m_bin_126[i] = m_bin_table.data.table[126][i];

                    for(int i=0; i<128; ++i)
                        m_bin_127[i] = m_bin_table.data.table[127][i];

                    for (uint64_t i = 0; i < 128; ++i)
                    {
                        __uint128_t class_cnt = m_bin_127[i];
                        if (class_cnt == 1)
                            m_space_for_bt[i] = 0;
                        else
                        {
                            size_t last = 0;
                            for (size_t i = 0; i < 127; ++i)
                            {
                                __uint128_t one = 1;
                                if (class_cnt & (one << i))
                                last = i;
                            }
                            m_space_for_bt[i] = last + 1;
                        }
                    }

                    for (size_t k = 0; k < 128; ++k)
                    {
                        std::fill(helper[k].begin(), helper[k].end(), 0);
                        __uint128_t total = 0;
                        for (size_t right_k = (k > 63) ? (k - 63) : 0;
                            right_k <= std::min(k, static_cast<size_t>(63)); ++right_k)
                        {
                            helper[k][right_k] = total;
                            total +=
                                m_bin_63[right_k] * m_bin_63[k - right_k];
                        }
                    }
                } // impl() end
        } iii;

    public:
        static inline uint32_t space_for_bt(uint32_t i)
        {
            return iii.m_space_for_bt[i];
        }

        static uint32_t sel(__uint128_t x, uint32_t i)
        {
            uint64_t hi = x >> 64;
            uint64_t lo = x;
            int cnt_hi = __builtin_popcountll(hi);
            int cnt_lo = __builtin_popcountll(lo);
            if (i >= cnt_lo + 1)
                return bits::sel(hi, i - cnt_lo) + 64;
            else
                return bits::sel(lo, i);
        }

        static inline __uint128_t nr_to_bin(uint8_t k, __uint128_t nr)
        {
        #ifndef NO_MY_OPT
            if (k == 127)
            {
                __uint128_t one = 1;
                return (one << 127) - 1;
            }
            else if (k == 0)
            {
                return 0;
            }
        #endif

            __uint128_t to_or = 0;
            constexpr __uint128_t one = 1;
            const bool first_bit = (nr >= (iii.m_bin_126[k]));
            if (first_bit)
            {
                nr -= iii.m_bin_126[k];
                --k;
                to_or |= one << 126;
            }

            const size_t right_k_from = (k > 63) ? (k - 63) : 0;
            const size_t right_k_to = std::min(k, static_cast<uint8_t>(63));

            size_t right_k = right_k_from;
            for (; right_k < right_k_to; ++right_k)
            {
                if (auto curr_nr = iii.helper[k][right_k + 1]; curr_nr >= nr)
                {
                    if (curr_nr == nr)
                        ++right_k;
                    break;
                }
            }

            nr -= iii.helper[k][right_k];

            const size_t left_k = k - right_k;

            const __uint128_t left_bin = iii.helper63.nr_to_bin(
                left_k, nr % iii.m_bin_63[left_k]);
            const __uint128_t right_bin = iii.helper63.nr_to_bin(
                right_k, nr / iii.m_bin_63[left_k]);

            return to_or | (left_bin << 63) | right_bin;
        }

        static inline __uint128_t bin_to_nr(const __uint128_t bin)
        {
        #ifndef NO_MY_OPT
            __uint128_t last = 1;
            last = last << 127;
            last -= 1;
            if ((bin == 0) || (bin == last))
            {
                return 0;
            }
        #endif
            const int k = popcountllll(bin);
            const __uint128_t one = 1;
            bool first_bit = bin & (one << 126);

            __uint128_t nr = first_bit * iii.m_bin_126[k];

            const uint64_t rem_k = k - first_bit;

            // 1073741823_10 = 111111111111111111111111111111_2
            const __uint128_t mask_low = 9223372036854775807ull;
            const uint64_t bin_left = (bin >> 63) & mask_low;
            const uint64_t bin_right = bin & mask_low;
            const int k_left = __builtin_popcountll(bin_left);
            const int k_right = __builtin_popcountll(bin_right);

            nr += iii.helper[rem_k][k_right];
            nr += iii.m_bin_63[k_left] *
                    static_cast<__uint128_t>(iii.helper63.bin_to_nr(bin_right));
            nr += iii.helper63.bin_to_nr(bin_left);
            return nr;
        }

        //! Decode the bit at position \f$ off \f$ of the block encoded by the pair
        //! (k, nr).
        static inline bool decode_bit(uint16_t k, number_type nr, uint16_t off)
        {
            return (nr_to_bin(k, nr) >> off) & static_cast<__uint128_t>(1);
        }

        static inline std::string toString(__uint128_t num)
        {
            std::string str;
            do
            {
                int digit = num % 10;
                str = std::to_string(digit) + str;
                num = (num - digit) / 10;
            } while (num != 0);
            return str;
        }

        static int popcountllll(__uint128_t n)
        {
            int cnt_hi = __builtin_popcountll(n >> 64);
            int cnt_lo = __builtin_popcountll(n);
            return cnt_hi + cnt_lo;
        }

                static inline __uint128_t sdsl_to_gcc(sdsl::uint128_t x)
        {
            return (static_cast<__uint128_t>(static_cast<uint64_t>(x >> 64))
                    << 64) + static_cast<uint64_t>(x);
        }

        static inline sdsl::uint128_t gcc_to_sdsl(__uint128_t x)
        {
            uint64_t nr_a = static_cast<uint64_t>(x);
            uint64_t nr_b = static_cast<uint64_t>(x >> 64);
            sdsl::uint128_t nr = nr_b;
            nr = nr << 64;
            nr += nr_a;
            return nr;
        }
};

}// end namespace sdsl

#endif