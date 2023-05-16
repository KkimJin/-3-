

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <stdbool.h>
#pragma warning(disable:4996)



#ifndef FIXED_H_
#define FIXED_H_

#include <ostream>
#include <exception>
#include <cstddef> // for size_t
#include <cstdint>
#include <type_traits>

#include <boost/operators.hpp>

namespace numeric {

    template <size_t I, size_t F>
    class Fixed;

    namespace detail {

        // helper templates to make magic with types :)
        // these allow us to determine resonable types from
        // a desired size, they also let us infer the next largest type
        // from a type which is nice for the division op
        template <size_t T>
        struct type_from_size {
            static const bool is_specialized = false;
            typedef void      value_type;
        };

#if defined(__GNUC__) && defined(__x86_64__)
        template <>
        struct type_from_size<128> {
            static const bool           is_specialized = true;
            static const size_t         size = 128;
            typedef __int128            value_type;
            typedef unsigned __int128   unsigned_type;
            typedef __int128            signed_type;
            typedef type_from_size<256> next_size;
        };
#endif

        template <>
        struct type_from_size<64> {
            static const bool           is_specialized = true;
            static const size_t         size = 64;
            typedef int64_t             value_type;
            typedef uint64_t            unsigned_type;
            typedef int64_t             signed_type;
            typedef type_from_size<128> next_size;
        };

        template <>
        struct type_from_size<32> {
            static const bool          is_specialized = true;
            static const size_t        size = 32;
            typedef int32_t            value_type;
            typedef uint32_t           unsigned_type;
            typedef int32_t            signed_type;
            typedef type_from_size<64> next_size;
        };

        template <>
        struct type_from_size<16> {
            static const bool          is_specialized = true;
            static const size_t        size = 16;
            typedef int16_t            value_type;
            typedef uint16_t           unsigned_type;
            typedef int16_t            signed_type;
            typedef type_from_size<32> next_size;
        };
        template <>
        struct type_from_size<15> {
            static const bool          is_specialized = true;
            static const size_t        size = 15;
            typedef int16_t            value_type;
            typedef uint16_t           unsigned_type;
            typedef int16_t            signed_type;
            typedef type_from_size<30> next_size;
        };
        template <>
        struct type_from_size<14> {
            static const bool          is_specialized = true;
            static const size_t        size = 14;
            typedef int16_t            value_type;
            typedef uint16_t           unsigned_type;
            typedef int16_t            signed_type;
            typedef type_from_size<28> next_size;
        };
        template <>
        struct type_from_size<13> {
            static const bool          is_specialized = true;
            static const size_t        size = 13;
            typedef int16_t            value_type;
            typedef uint16_t           unsigned_type;
            typedef int16_t            signed_type;
            typedef type_from_size<26> next_size;
        };
        template <>
        struct type_from_size<12> {
            static const bool          is_specialized = true;
            static const size_t        size = 12;
            typedef int8_t             value_type;
            typedef uint8_t            unsigned_type;
            typedef int8_t             signed_type;
            typedef type_from_size<24> next_size;
        };
        template <>
        struct type_from_size<11> {
            static const bool          is_specialized = true;
            static const size_t        size = 11;
            typedef int8_t             value_type;
            typedef uint8_t            unsigned_type;
            typedef int8_t             signed_type;
            typedef type_from_size<22> next_size;
        };

        template <>
        struct type_from_size<10> {
            static const bool          is_specialized = true;
            static const size_t        size = 10;
            typedef int8_t             value_type;
            typedef uint8_t            unsigned_type;
            typedef int8_t             signed_type;
            typedef type_from_size<20> next_size;
        };

        template <>
        struct type_from_size<9> {
            static const bool          is_specialized = true;
            static const size_t        size = 9;
            typedef int8_t             value_type;
            typedef uint8_t            unsigned_type;
            typedef int8_t             signed_type;
            typedef type_from_size<18> next_size;
        };

        template <>
        struct type_from_size<8> {
            static const bool          is_specialized = true;
            static const size_t        size = 8;
            typedef int8_t             value_type;
            typedef uint8_t            unsigned_type;
            typedef int8_t             signed_type;
            typedef type_from_size<16> next_size;
        };

        // this is to assist in adding support for non-native base
        // types (for adding big-int support), this should be fine
        // unless your bit-int class doesn't nicely support casting
        template <class B, class N>
        B next_to_base(const N& rhs) {
            return static_cast<B>(rhs);
        }

        struct divide_by_zero : std::exception {
        };

        template <size_t I, size_t F>
        Fixed<I, F> divide(const Fixed<I, F>& numerator, const Fixed<I, F>& denominator, Fixed<I, F>& remainder, typename std::enable_if<type_from_size<I + F>::next_size::is_specialized>::type* = 0) {

            typedef typename Fixed<I, F>::next_type next_type;
            typedef typename Fixed<I, F>::base_type base_type;
            static const size_t fractional_bits = Fixed<I, F>::fractional_bits;

            next_type t(numerator.to_raw());
            t <<= fractional_bits;

            Fixed<I, F> quotient;

            quotient = Fixed<I, F>::from_base(next_to_base<base_type>(t / denominator.to_raw()));
            remainder = Fixed<I, F>::from_base(next_to_base<base_type>(t % denominator.to_raw()));

            return quotient;
        }

        template <size_t I, size_t F>
        Fixed<I, F> divide(Fixed<I, F> numerator, Fixed<I, F> denominator, Fixed<I, F>& remainder, typename std::enable_if<!type_from_size<I + F>::next_size::is_specialized>::type* = 0) {

            // NOTE(eteran): division is broken for large types :-(
            // especially when dealing with negative quantities

            typedef typename Fixed<I, F>::base_type     base_type;
            typedef typename Fixed<I, F>::unsigned_type unsigned_type;

            static const int bits = Fixed<I, F>::total_bits;

            if (denominator == 0) {
                throw divide_by_zero();
            }
            else {

                int sign = 0;

                Fixed<I, F> quotient;

                if (numerator < 0) {
                    sign ^= 1;
                    numerator = -numerator;
                }

                if (denominator < 0) {
                    sign ^= 1;
                    denominator = -denominator;
                }

                base_type n = numerator.to_raw();
                base_type d = denominator.to_raw();
                base_type x = 1;
                base_type answer = 0;

                // egyptian division algorithm
                while ((n >= d) && (((d >> (bits - 1)) & 1) == 0)) {
                    x <<= 1;
                    d <<= 1;
                }

                while (x != 0) {
                    if (n >= d) {
                        n -= d;
                        answer += x;
                    }

                    x >>= 1;
                    d >>= 1;
                }

                unsigned_type l1 = n;
                unsigned_type l2 = denominator.to_raw();

                // calculate the lower bits (needs to be unsigned)
                // unfortunately for many fractions this overflows the type still :-/
                const unsigned_type lo = (static_cast<unsigned_type>(n) << F) / denominator.to_raw();

                quotient = Fixed<I, F>::from_base((answer << F) | lo);
                remainder = n;

                if (sign) {
                    quotient = -quotient;
                }

                return quotient;
            }
        }

        // this is the usual implementation of multiplication
        template <size_t I, size_t F>
        void multiply(const Fixed<I, F>& lhs, const Fixed<I, F>& rhs, Fixed<I, F>& result, typename std::enable_if<type_from_size<I + F>::next_size::is_specialized>::type* = 0) {

            typedef typename Fixed<I, F>::next_type next_type;
            typedef typename Fixed<I, F>::base_type base_type;

            static const size_t fractional_bits = Fixed<I, F>::fractional_bits;

            next_type t(static_cast<next_type>(lhs.to_raw()) * static_cast<next_type>(rhs.to_raw()));
            t >>= fractional_bits;
            result = Fixed<I, F>::from_base(next_to_base<base_type>(t));
        }

        // this is the fall back version we use when we don't have a next size
        // it is slightly slower, but is more robust since it doesn't
        // require and upgraded type
        template <size_t I, size_t F>
        void multiply(const Fixed<I, F>& lhs, const Fixed<I, F>& rhs, Fixed<I, F>& result, typename std::enable_if<!type_from_size<I + F>::next_size::is_specialized>::type* = 0) {

            typedef typename Fixed<I, F>::base_type base_type;

            static const size_t fractional_bits = Fixed<I, F>::fractional_bits;
            static const size_t integer_mask = Fixed<I, F>::integer_mask;
            static const size_t fractional_mask = Fixed<I, F>::fractional_mask;

            // more costly but doesn't need a larger type
            const base_type a_hi = (lhs.to_raw() & integer_mask) >> fractional_bits;
            const base_type b_hi = (rhs.to_raw() & integer_mask) >> fractional_bits;
            const base_type a_lo = (lhs.to_raw() & fractional_mask);
            const base_type b_lo = (rhs.to_raw() & fractional_mask);

            const base_type x1 = a_hi * b_hi;
            const base_type x2 = a_hi * b_lo;
            const base_type x3 = a_lo * b_hi;
            const base_type x4 = a_lo * b_lo;

            result = Fixed<I, F>::from_base((x1 << fractional_bits) + (x3 + x2) + (x4 >> fractional_bits));

        }
    }

    /*
     * inheriting from boost::operators enables us to be a drop in replacement for base types
     * without having to specify all the different versions of operators manually
     */
    template <size_t I, size_t F>
    class Fixed : boost::operators<Fixed<I, F>> {
        static_assert(detail::type_from_size<I + F>::is_specialized, "invalid combination of sizes");

    public:
        static const size_t fractional_bits = F;
        static const size_t integer_bits = I;
        static const size_t total_bits = I + F;

        typedef detail::type_from_size<total_bits>             base_type_info;

        typedef typename base_type_info::value_type            base_type;
        typedef typename base_type_info::next_size::value_type next_type;
        typedef typename base_type_info::unsigned_type         unsigned_type;

    public:
        static const size_t base_size = base_type_info::size;
        static const base_type fractional_mask = ~((~base_type(0)) << fractional_bits);
        static const base_type integer_mask = ~fractional_mask;

    public:
        static const base_type one = base_type(1) << fractional_bits;

    public: // constructors
        Fixed() : data_(0) {
        }

        Fixed(long n) : data_(base_type(n) << fractional_bits) {
            // TODO(eteran): assert in range!
        }

        Fixed(unsigned long n) : data_(base_type(n) << fractional_bits) {
            // TODO(eteran): assert in range!
        }

        Fixed(int n) : data_(base_type(n) << fractional_bits) {
            // TODO(eteran): assert in range!
        }

        Fixed(unsigned int n) : data_(base_type(n) << fractional_bits) {
            // TODO(eteran): assert in range!
        }

        Fixed(float n) : data_(static_cast<base_type>(n* one)) {
            // TODO(eteran): assert in range!
        }

        Fixed(double n) : data_(static_cast<base_type>(n* one)) {
            // TODO(eteran): assert in range!
        }

        Fixed(const Fixed& o) : data_(o.data_) {
        }

        Fixed& operator=(const Fixed& o) {
            data_ = o.data_;
            return *this;
        }

    private:
        // this makes it simpler to create a fixed point object from
        // a native type without scaling
        // use "Fixed::from_base" in order to perform this.
        struct NoScale {};

        Fixed(base_type n, const NoScale&) : data_(n) {
        }

    public:
        static Fixed from_base(base_type n) {
            return Fixed(n, NoScale());
        }

    public: // comparison operators
        bool operator==(const Fixed& o) const {
            return data_ == o.data_;
        }

        bool operator<(const Fixed& o) const {
            return data_ < o.data_;
        }

    public: // unary operators
        bool operator!() const {
            return !data_;
        }

        Fixed operator~() const {
            Fixed t(*this);
            t.data_ = ~t.data_;
            return t;
        }

        Fixed operator-() const {
            Fixed t(*this);
            t.data_ = -t.data_;
            return t;
        }

        Fixed operator+() const {
            return *this;
        }

        Fixed& operator++() {
            data_ += one;
            return *this;
        }

        Fixed& operator--() {
            data_ -= one;
            return *this;
        }

    public: // basic math operators
        Fixed& operator+=(const Fixed& n) {
            data_ += n.data_;
            return *this;
        }

        Fixed& operator-=(const Fixed& n) {
            data_ -= n.data_;
            return *this;
        }

        Fixed& operator&=(const Fixed& n) {
            data_ &= n.data_;
            return *this;
        }

        Fixed& operator|=(const Fixed& n) {
            data_ |= n.data_;
            return *this;
        }

        Fixed& operator^=(const Fixed& n) {
            data_ ^= n.data_;
            return *this;
        }

        Fixed& operator*=(const Fixed& n) {
            detail::multiply(*this, n, *this);
            return *this;
        }

        Fixed& operator/=(const Fixed& n) {
            Fixed temp;
            *this = detail::divide(*this, n, temp);
            return *this;
        }

        Fixed& operator>>=(const Fixed& n) {
            data_ >>= n.to_int();
            return *this;
        }

        Fixed& operator<<=(const Fixed& n) {
            data_ <<= n.to_int();
            return *this;
        }

    public: // conversion to basic types
        int to_int() const {
            return (data_ & integer_mask) >> fractional_bits;
        }

        unsigned int to_uint() const {
            return (data_ & integer_mask) >> fractional_bits;
        }

        float to_float() const {
            return static_cast<float>(data_) / Fixed::one;
        }

        double to_double() const {
            return static_cast<double>(data_) / Fixed::one;
        }

        base_type to_raw() const {
            return data_;
        }

    public:
        void swap(Fixed& rhs) {
            using std::swap;
            swap(data_, rhs.data_);
        }

    public:
        base_type data_;
    };

    // if we have the same fractional portion, but differing integer portions, we trivially upgrade the smaller type
    template <size_t I1, size_t I2, size_t F>
    typename std::conditional<I1 >= I2, Fixed<I1, F>, Fixed<I2, F>>::type operator+(const Fixed<I1, F>& lhs, const Fixed<I2, F>& rhs) {

        typedef typename std::conditional<
            I1 >= I2,
            Fixed<I1, F>,
            Fixed<I2, F>
        >::type T;

        const T l = T::from_base(lhs.to_raw());
        const T r = T::from_base(rhs.to_raw());
        return l + r;
    }

    template <size_t I1, size_t I2, size_t F>
    typename std::conditional<I1 >= I2, Fixed<I1, F>, Fixed<I2, F>>::type operator-(const Fixed<I1, F>& lhs, const Fixed<I2, F>& rhs) {

        typedef typename std::conditional<
            I1 >= I2,
            Fixed<I1, F>,
            Fixed<I2, F>
        >::type T;

        const T l = T::from_base(lhs.to_raw());
        const T r = T::from_base(rhs.to_raw());
        return l - r;
    }

    template <size_t I1, size_t I2, size_t F>
    typename std::conditional<I1 >= I2, Fixed<I1, F>, Fixed<I2, F>>::type operator*(const Fixed<I1, F>& lhs, const Fixed<I2, F>& rhs) {

        typedef typename std::conditional<
            I1 >= I2,
            Fixed<I1, F>,
            Fixed<I2, F>
        >::type T;

        const T l = T::from_base(lhs.to_raw());
        const T r = T::from_base(rhs.to_raw());
        return l * r;
    }

    template <size_t I1, size_t I2, size_t F>
    typename std::conditional<I1 >= I2, Fixed<I1, F>, Fixed<I2, F>>::type operator/(const Fixed<I1, F>& lhs, const Fixed<I2, F>& rhs) {

        typedef typename std::conditional<
            I1 >= I2,
            Fixed<I1, F>,
            Fixed<I2, F>
        >::type T;

        const T l = T::from_base(lhs.to_raw());
        const T r = T::from_base(rhs.to_raw());
        return l / r;
    }

    template <size_t I, size_t F>
    std::ostream& operator<<(std::ostream& os, const Fixed<I, F>& f) {
        os << f.to_double();
        return os;
    }

    template <size_t I, size_t F>
    const size_t Fixed<I, F>::fractional_bits;

    template <size_t I, size_t F>
    const size_t Fixed<I, F>::integer_bits;

    template <size_t I, size_t F>
    const size_t Fixed<I, F>::total_bits;

}

#endif
using namespace numeric;
using namespace cv;
using namespace std;
typedef unsigned char BYTE;

int PL[720][1280][2];
int ons[720][1280][2];
int offs[720][1280][2];
int fps = 30;
int width = 1280;
int height = 720;
int Ncell = 1280 * 720;
int Np = 1;
int Nsp = 6;
int Nts = 4;
int tau_sfa = 1000;
float time_interval = (int)(1000 / fps);
Fixed<32, 32> hp_sfa = tau_sfa / (tau_sfa + time_interval);
int Tpm = 10;
int Cspi = 4;
float W_on = 0.5;
float W_off = 1;
float W_onoff = 1;
float Tsp = 0.609375f;
float Tsfa = 0.00390625f;
Fixed<32, 32> W_i_on = 1.0;
float W_i_on_base = 1.0;
Fixed<32, 32> W_i_off = 0.5f;
float W_i_off_base = 0.5f;
float dc = 0.125f;
float Csig = 0.5;
Fixed<32, 32> W_g[3][3];
float W_gb[3][3];
int Inh_ON[720][1280];
int Inh_OFF[720][1280];
float tau_ON[3];
int Ts = 6;
float tau_OFF[3];
float tau_PM = 90;
Fixed<32, 32> lp_ON[3];
float lpb_ON[3];
Fixed<32, 32> lp_OFF[3];
float lpb_OFF[3];
Fixed<32, 32> lp_PM = time_interval / (time_interval + tau_PM);
int Cw = 4;
int S_on;
int S_off;
int scells[720][1280];
int gcells[720][1280];
int pm[2];
int mp[2];
Fixed<32, 32> smp[2];
Fixed<32, 32> sfa[2];
int spike[250];
int collision[250] = {};
int s_cells[720][1280];

int HighpassFilter(byte pre_input, byte cur_input)
{
    return cur_input - pre_input;
}

float LowpassFilter(float cur_input, float pre_input, float lp_t)
{
    return lp_t * cur_input + (1 - lp_t) * pre_input;
}
float Convolution(int x, int y, int inputMatrix[720][1280][2], float kernel[3][3], int cur_frame, int pre_frame, float lp_delay[3])
{
    float tmp = 0;
    int r, c;
    float lp;
    for (int i = -Np; i < Np + 1; i++)
    {
        //check border
        r = x + i;
        while (r < 0)
            r += 1;
        while (r >= height)
            r -= 1;
        for (int j = -Np; j < Np + 1; j++)
        {
            //check border
            c = y + j;
            while (c < 0)
                c += 1;
            while (c >= width)
                c -= 1;
            //centre cell
            if (i == 0 && j == 0)
                lp = LowpassFilter(inputMatrix[r][c][cur_frame], inputMatrix[r][c][pre_frame], lp_delay[0]);
            //nearest cells
            else if (i == 0 || j == 0)
                lp = LowpassFilter(inputMatrix[r][c][cur_frame], inputMatrix[r][c][pre_frame], lp_delay[1]);
            //diagonal cells
            else
                lp = LowpassFilter(inputMatrix[r][c][cur_frame], inputMatrix[r][c][pre_frame], lp_delay[2]);
            tmp += lp * kernel[i + Np][j + Np];
        }
    }
    return tmp;
}


int SupralinearSummation(int on_exc, int off_exc)
{
    return W_on * on_exc + W_off * off_exc + W_onoff * on_exc * off_exc;
}
int sCellValue(int exc, int inh, float wi)
{
    float tmp = exc - inh * wi;
    if (tmp <= 0)
        return 0;
    else
        return tmp;
}
float Convolving(int x, int y, int matrix[720][1280], float kernel[3][3])
{
    float tmp = 0;
    int r;
    int c;
    for (int i = -Np; i < Np + 1; i++)
    {
        r = x + i;
        while (r < 0) {
            r += 1;
        }
        while (r >= height) {
            r -= 1;
        }
        for (int j = -Np; j < Np + 1; j++)
        {
            c = y + j;
            while (c < 0)
                c += 1;
            while (c >= 1280)
                c -= 1;
            tmp = tmp + (matrix[r][c] * kernel[i + Np][j + Np]);
        }
    }
    return tmp;
}
float SigmoidTransfer(int Kf)
{
    return (float)pow(1 + exp(-Kf * pow(Ncell * Csig, -1)), -1);
}

float SFA_HPF(float pre_sfa, float pre_mp, float cur_mp)
{
    float diff_mp = cur_mp - pre_mp;
    if (diff_mp <= Tsfa)
    {
        float tmp_mp = hp_sfa.to_float() * (pre_sfa + diff_mp);
        if (tmp_mp < 0.5f)
            return 0.5f;
        else
            return tmp_mp;
    }
    else
    {
        float tmp_mp = hp_sfa.to_float() * cur_mp;
        if (tmp_mp < 0.5f)
            return 0.5f;
        else
            return tmp_mp;
    }
}
int Spiking(float sfa)
{
    int spi = (int)floor(exp(Cspi * (sfa - Tsp)));
    if (spi == 0)
        return 0;
    else
        return spi;

}
float Conv_ON[3][3] =
{ 0.25f,0.5f,0.25f,0.5f,2.0,0.5f,0.25f,0.5f,0.25f };


float Conv_OFF[3][3] =
{ 0.125f,0.25f,0.125f,0.25f,1.0,0.25f,0.125f,0.25f,0.125f };


int main() {

    FILE* fp1 = fopen("data.txt", "w");
    FILE* fp2 = fopen("spike.txt", "w");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            W_g[i][j] = (1.0 / 9.0);
            W_gb[i][j] = W_g[i][j].to_float();
        }
    }
    for (int i = 0; i < 3; i++) {
        tau_ON[i] = 15 + i * 15;
        tau_OFF[i] = 60 + i * 60;
        lp_ON[i] = time_interval / (time_interval + tau_ON[i]);
        lpb_ON[i] = lp_ON[i].to_float();
        lp_OFF[i] = time_interval / (time_interval + tau_OFF[i]);
        lpb_OFF[i] = lp_OFF[i].to_float();

    }
    for (int i = 2; i < 43; i++) {

        int pre_frame = (i - 1) % 2;
        int cur_frame = (i - 2) % 2;
        int tmp_pm = 0;
        float tmp_sum = 0;
        float scale;
        float max = 0;
        char filename1[75];
        sprintf(filename1, "D:\\lgmd2_test\\Fig11\\2\\frame%d.bmp", i - 1);
        Mat img1 = imread(filename1);
        Mat gray1;
        cvtColor(img1, gray1, COLOR_RGB2GRAY);
        char filename2[75];
        sprintf(filename2, "D:\\lgmd2_test\\Fig11\\2\\frame%d.bmp", i);
        Mat img2 = imread(filename2);
        Mat gray2;
        cvtColor(img2, gray2, COLOR_RGB2GRAY);
        Mat gray3;
        cvtColor(img2, gray3, COLOR_RGB2GRAY);
        Mat test1 = gray1;
        Mat test2 = gray2;
        Mat test3 = gray3;
        char filename3[75];
        char filename4[75];
        char filename5[75];
        for (int y = 0; y < gray1.rows; y++)
        {
            for (int x = 0; x < gray1.cols; x++)
            {

                if (i == 2) {
                    PL[y][x][cur_frame] = gray2.at<uchar>(y, x) - gray1.at<uchar>(y, x);
                    ons[y][x][cur_frame] = max(0, PL[y][x][cur_frame]);
                    offs[y][x][cur_frame] = -min(PL[y][x][cur_frame], 0);
                    tmp_pm += abs(PL[y][x][cur_frame]);
                    test1.at<uchar>(y, x) = ons[y][x][cur_frame];
                    test2.at<uchar>(y, x) = offs[y][x][cur_frame];
                }
                else {
                    PL[y][x][cur_frame] = gray2.at<uchar>(y, x) - gray1.at<uchar>(y, x) + PL[y][x][pre_frame] / (1 + exp(1));
                    ons[y][x][cur_frame] = max(0, PL[y][x][cur_frame]) + dc * ons[y][x][pre_frame];
                    offs[y][x][cur_frame] = -min(PL[y][x][cur_frame], 0) + dc * offs[y][x][pre_frame];
                    tmp_pm += abs(PL[y][x][cur_frame]);
                    test1.at<uchar>(y, x) = ons[y][x][cur_frame];
                    test2.at<uchar>(y, x) = offs[y][x][cur_frame];
                }

            }
        }
        sprintf(filename3, "D:\\lgmd2_test\\Fig11\\Pon-layer\\Pon-layer%d.bmp", i - 1);
        imwrite(filename3, test1);
        sprintf(filename4, "D:\\lgmd2_test\\Fig11\\Poff-layer\\Poff-layer%d.bmp", i - 1);
        imwrite(filename4, test2);

        if (i == 2) {
            pm[cur_frame] = tmp_pm / Ncell;
        }
        else {
            pm[cur_frame] = tmp_pm / Ncell;
            pm[cur_frame] = LowpassFilter(pm[cur_frame], pm[pre_frame], lp_PM.to_float());
            W_i_off = pm[pre_frame] / Tpm;
            if (W_i_off.to_float() < W_i_off_base)
                W_i_off = W_i_off_base;
            W_i_on = pm[pre_frame] / Tpm;
            if (W_i_on.to_float() < W_i_on_base)
                W_i_on = W_i_on_base;
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (i == 2) {
                    Inh_ON[y][x] = 0;
                    Inh_OFF[y][x] = 0;
                    scells[y][x] = 0;
                    if (scells[y][x] >= Ts)
                        s_cells[y][x] = scells[y][x];

                    else
                        s_cells[y][x] = 0;
                    test1.at<uchar>(y, x) = (uchar)Inh_ON[y][x];
                    test2.at<uchar>(y, x) = (uchar)Inh_OFF[y][x];
                    test3.at<uchar>(y, x) = (uchar)scells[y][x];
                }
                else {
                    Inh_ON[y][x] = (int)Convolution(y, x, ons, Conv_ON, cur_frame, pre_frame, lpb_ON);
                    Inh_OFF[y][x] = (int)Convolution(y, x, offs, Conv_OFF, cur_frame, pre_frame, lpb_OFF);
                    S_on = sCellValue(ons[y][x][cur_frame], Inh_ON[y][x], W_i_on.to_float());
                    S_off = sCellValue(offs[y][x][cur_frame], Inh_OFF[y][x], W_i_off.to_float());
                    scells[y][x] = (int)SupralinearSummation(S_on, S_off);
                    if (scells[y][x] >= Ts) {
                        s_cells[y][x] = scells[y][x];
                    }
                    else
                        s_cells[y][x] = 0;
                    test1.at<uchar>(y, x) = (uchar)Inh_ON[y][x];
                    test2.at<uchar>(y, x) = (uchar)Inh_OFF[y][x];
                    test3.at<uchar>(y, x) = (uchar)scells[y][x];
                }
            }
        }
        sprintf(filename3, "D:\\lgmd2_test\\Fig11\\Ion-layer\\Ion-layer%d.bmp", i - 1);
        imwrite(filename3, test1);
        sprintf(filename4, "D:\\lgmd2_test\\Fig11\\Ioff-layer\\Ioff-layer%d.bmp", i - 1);
        imwrite(filename4, test2);
        sprintf(filename5, "D:\\lgmd2_test\\Fig11\\S-layer\\S-layer%d.bmp", i - 1);
        imwrite(filename5, test3);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                gcells[y][x] = (int)Convolving(y, x, s_cells, W_gb);
                tmp_sum += (int)gcells[y][x];
                test1.at<uchar>(y, x) = gcells[y][x];
            }
        }


        sprintf(filename3, "D:\\lgmd2_test\\Fig11\\G-layer\\G-layer%d.bmp", i - 1);
        imwrite(filename3, test1);
        //waitKey(0);
        //imshow("img", test);
        mp[cur_frame] = tmp_sum;
        smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);

        if (i > 2) {
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame].to_float(), smp[pre_frame].to_float(), smp[cur_frame].to_float());
        }
        else {
            sfa[cur_frame] = 0.5f;
        }


        spike[i - 2] = Spiking(sfa[cur_frame].to_float());
        fprintf(fp1, "%d %f\n", i - 2, sfa[cur_frame].to_float());
        fprintf(fp2, "%d %d\n", i - 2, spike[i - 2]);



    }
    fclose(fp1);
    fclose(fp2);

    for (int t = 0; t < 138; t++) {
        int sum_spi = 0;
        int detect = 0;
        for (int i = t; i <= t + Nts; i++) {
            sum_spi += spike[i];
        }
        if (sum_spi >= Nsp) {
            collision[t + Nts] = 1;
            printf("%d번째 프레임에서 충돌이 감지됩니다.\n", t + Nts);

        }
    }
    return 0;
}