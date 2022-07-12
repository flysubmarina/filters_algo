#include <stdint.h>
#include <cstdio>
#include <complex>
#define _USE_MATH_DEFINES

enum ImageType
{
    PNG,
    JPG,
    BMP,
    TGA
};

struct Image
{
    uint8_t *data = NULL;
    size_t size = 0;
    int w;
    int h;
    int channels;

    Image(const char *filename);
    Image(int w, int h, int channels);
    Image(const Image &image);

    ~Image();

    bool read(const char *filename);
    bool write(const char *filename);

    ImageType getImageType(const char *filename);

    Image &grayscale_avg();
    Image &grayscale_lum();

    Image &std_convolve_claim_to_0(uint8_t channel, uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc); // standart convolve for channel;

    Image &std_convolve(uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc); // standart convolve for each channel;

    Image &fd_convolve_claim_to_0(uint8_t channel, uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc); // furrie convolve for channel;

    Image &fd_convolve(uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc); // furrie convolve for each channel;

    static uint32_t rev(uint32_t n, uint32_t a);
    static void bit_rev(uint32_t n, std::complex<double> a[], std::complex<double> *A);

    static void fft(uint32_t n, std::complex<double> x[], std::complex<double> *X);
    static void ifft(uint32_t n, std::complex<double> X[], std::complex<double> *x);

    static void dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double> *X);
    static void idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double> *x);

    static inline void pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double> *p);
    static void pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double> *pad_ker);

    static double *getGaussianKernel(int radius);
};