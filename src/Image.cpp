#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _USE_MATH_DEFINES

#include "Image.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <math.h>

#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)

double byte_bound(double value)
{
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

Image::Image(const char *filename)
{
    if (read(filename))
    {
        printf("Read %s \n", filename);
        size = w * h * channels;
    }
    else
    {
        printf("Failed to read %s \n", filename);
    }
}
Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels)
{
    size = w * h * channels;
    data = new uint8_t[size];
}
Image::Image(const Image &img) : Image(img.w, img.h, img.channels)
{
    memcpy(data, img.data, img.size);
}

Image::~Image()
{
    stbi_image_free(data);
}

bool Image::read(const char *filename)
{
    data = stbi_load(filename, &w, &h, &channels, 0);

    return data != NULL;
}

uint32_t Image::rev(uint32_t n, uint32_t a)
{
    uint8_t max_bits = (uint8_t)ceil(log2(n));

    uint32_t revesed_a = 0;
    for (uint8_t i = 0; i < max_bits; ++i)
    {
        if (a & (1 << i))
        {
            revesed_a = revesed_a | (1 << (max_bits - 1 - i));
        }
    }

    return revesed_a;
}

void Image::bit_rev(uint32_t n, std::complex<double> a[], std::complex<double> *A)
{
    for (uint8_t i = 0; i < n; ++i)
    {
        A[rev(n, i)] = a[i];
    }
}

void Image::fft(uint32_t n, std::complex<double> x[], std::complex<double> *X)
{
    if (x != X)
    {
        memcpy(X, x, sizeof(std::complex<double>) * n);
    }

    uint32_t sub_probs = 1;
    uint32_t sub_prob_size = n;
    uint32_t half;
    uint32_t i;
    uint32_t j;
    uint32_t j_begin;
    uint32_t j_end;

    std::complex<double> w_step;
    std::complex<double> w;
    std::complex<double> tmp1, tmp2;

    while (sub_prob_size > 1)
    {
        half = sub_prob_size >> 1;
        w_step = std::complex<double>(cos(-2 * M_PI / sub_prob_size), sin(-2 * M_PI / sub_prob_size));

        for (i = 0; i < sub_probs; ++i)
        {
            j_begin = i * sub_prob_size;
            j_end = j_begin + half;
            w = std::complex<double>(1, 0);
            for (j = j_begin; j < j_end; ++j)
            {
                tmp1 = X[j];
                tmp2 = X[j + half];
                X[j] = tmp1 + tmp2;
                X[j + half] = (tmp1 - tmp2) * w;
                w *= w_step;
            }
        }

        sub_probs <<= 1;
        sub_prob_size = half;
    }
}
void Image::ifft(uint32_t n, std::complex<double> X[], std::complex<double> *x)
{
    if (X != x)
    {
        memcpy(x, X, sizeof(std::complex<double>) * n);
    }

    uint32_t sub_probs = n >> 1;
    uint32_t sub_prob_size;
    uint32_t half = 1;
    uint32_t i;
    uint32_t j;
    uint32_t j_begin;
    uint32_t j_end;

    std::complex<double> w_step;
    std::complex<double> w;
    std::complex<double> tmp1, tmp2;

    while (half < n)
    {
        sub_prob_size = half << 1;
        w_step = std::complex<double>(cos(2 * M_PI / sub_prob_size), sin(2 * M_PI / sub_prob_size));

        for (i = 0; i < sub_probs; ++i)
        {
            j_begin = i * sub_prob_size;
            j_end = j_begin + half;
            w = std::complex<double>(1, 0);
            for (j = j_begin; j < j_end; ++j)
            {
                tmp1 = x[j];
                tmp2 = w * x[j + half];
                x[j] = tmp1 + tmp2;
                x[j + half] = tmp1 - tmp2;
                w *= w_step;
            }
        }

        sub_probs >>= 1;
        half = sub_prob_size;
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        x[i] /= n;
    }
}

void Image::dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double> *X)
{
    std::complex<double> *intermidiate = new std::complex<double>[m * n];
    for (uint32_t i = 0; i < m; ++i)
    {
        fft(n, x + i * n, intermidiate + i * n);
    }

    for (uint32_t j = 0; j < n; ++j)
    {
        for (uint32_t i = 0; i < m; ++i)
        {
            X[j * m + i] = intermidiate[i * n + j];
        }
        fft(m, X + j * m, X + j * m);
    }

    delete[] intermidiate;
}
void Image::idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double> *x)
{
    std::complex<double> *intermidiate = new std::complex<double>[m * n];

    for (uint32_t j = 0; j < n; ++j)
    {
        ifft(m, X + j * m, intermidiate + j * m);
    }

    for (uint32_t i = 0; i < m; ++i)
    {
        for (uint32_t j = 0; j < n; ++j)
        {
            x[i * n + j] = intermidiate[j * m + i];
        }

        ifft(n, x + i * n, x + i * n);
    }

    delete[] intermidiate;
}

void Image::pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double> *p)
{
    for (uint64_t i = 0; i < l; ++i)
    {
        p[i] = a[i] * b[i];
    }
}

void Image::pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double> *pad_ker)
{
    for (long i = -((long)cr); i < (long)ker_h - cr; ++i)
    {
        uint32_t r = i < 0 ? i + ph : i;
        for (long j = -((long)cc); j < (long)ker_w - cc; ++j)
        {
            uint32_t c = j < 0 ? j + pw : j;
            pad_ker[r * pw + c] = std::complex<double>(ker[(i + cr) * ker_w + (j + cc)], 0);
        }
    }
}

Image &Image::std_convolve(uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc)
{
    std_convolve_claim_to_0(0, ker_w, ker_h, ker, cr, cc);
    std_convolve_claim_to_0(1, ker_w, ker_h, ker, cr, cc);
    std_convolve_claim_to_0(2, ker_w, ker_h, ker, cr, cc);

    return *this;
}
Image &Image::fd_convolve(uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc)
{
    fd_convolve_claim_to_0(0, ker_w, ker_h, ker, cr, cc);
    fd_convolve_claim_to_0(1, ker_w, ker_h, ker, cr, cc);
    fd_convolve_claim_to_0(2, ker_w, ker_h, ker, cr, cc);

    return *this;
}

Image &Image::grayscale_avg()
{
    if (channels < 3)
    {
        printf("Image %p has less than 3 channels, it is assumed to already be grayscale", this);
    }
    else
    {
        for (int i = 0; i < size; i += channels)
        {
            //(r+g+b)/3
            int gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
            memset(data + i, gray, 3);
        }
    }
    return *this;
}

Image &Image::grayscale_lum()
{
    if (channels < 3)
    {
        printf("Image %p has less than 3 channels, it is assumed to already be grayscale.", this);
    }
    else
    {
        for (int i = 0; i < size; i += channels)
        {
            int gray = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
            memset(data + i, gray, 3);
        }
    }
    return *this;
}

bool Image::write(const char *filename)
{
    ImageType type = getImageType(filename);
    int success;
    switch (type)
    {
    case PNG:
        success = stbi_write_png(filename, w, h, channels, data, w * channels);
        break;
    case JPG:
        success = stbi_write_jpg(filename, w, h, channels, data, 100);
        break;
    case BMP:
        success = stbi_write_bmp(filename, w, h, channels, data);
        break;
    case TGA:
        success = stbi_write_tga(filename, w, h, channels, data);
        break;
    }

    return success != 0;
}

// double[] Image::getMaskWithCoff(double[] ker, uint64_t ker_size, double coff)
// {
//     double[] new_ker[ker_size];

//     for (uint64_t i = 0; i < ker_size; ++i)
//     {
//         new_ker[i] = ker[i] / coff;
//     }

//     return new_ker;
// }

ImageType Image::getImageType(const char *filename)
{
    const char *ext = strrchr(filename, '.');

    if (ext != nullptr)
    {
        if (strcmp(ext, ".png") == 0)
        {
            return PNG;
        }
        else if (strcmp(ext, ".jpg") == 0)
        {
            return JPG;
        }
        else if (strcmp(ext, ".tga") == 0)
        {
            return TGA;
        }
        else if (strcmp(ext, ".bmp") == 0)
        {
            return BMP;
        }
    }

    return PNG;
}
Image &Image::fd_convolve_claim_to_0(uint8_t channel, uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc)
{
    uint32_t pw = 1 << ((uint32_t)ceil(log2(w + ker_w - 1)));
    uint32_t ph = 1 << ((uint32_t)ceil(log2(h + ker_h - 1)));
    uint64_t psize = pw * ph;

    std::complex<double> *pad_img = new std::complex<double>[psize];

    for (uint32_t i = 0; i < h; ++i)
    {
        for (uint32_t j = 0; j < w; ++j)
        {
            pad_img[i * pw + j] = std::complex<double>(data[(i * w + j) * channels + channel], 0);
        }
    }

    std::complex<double> *pad_ker = new std::complex<double>[psize];
    pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

    //convolution
    dft_2D(ph, pw, pad_img, pad_img);
    dft_2D(ph, pw, pad_ker, pad_ker);

    // //multiply kernel and image which padded before
    pointwise_product(psize, pad_img, pad_ker, pad_img);

    idft_2D(ph, pw, pad_img, pad_img);

    for (uint32_t i = 0; i < h; ++i)
    {
        for (uint32_t j = 0; j < w; ++j)
        {
            data[(i * w + j) * channels + channel] = byte_bound((uint8_t)round(pad_img[i * pw + j].real()));
        }
    }

    return *this;
}

Image &Image::std_convolve_claim_to_0(uint8_t channel, uint8_t ker_w, uint8_t ker_h, double ker[], uint32_t cr, uint32_t cc)
{
    uint8_t new_data[w * h];
    uint64_t center = cr * ker_w + cc;

    for (uint64_t k = channel; k < size; k += channels)
    {
        double c = 0;

        for (long i = -((long)cr); i < (long)ker_h - cr; ++i)
        {
            long row = ((long)k / channels) / w - i;

            if (row < 0 || row > h - 1)
            {
                continue;
            }

            for (long j = -((long)cc); j < (long)ker_w - cc; ++j)
            {
                long col = ((long)k / channels) % w - j;

                c += ker[center + i * (long)ker_w + j] * data[(row * w + col) * channels + channel];

                // printf("value is = %f", ker[center + i * (long)ker_w + j]);
            }
        }

        new_data[k / channels] = (uint8_t)byte_bound(round(c));
    }

    for (uint64_t k = channel; k < size; k += channels)
    {
        data[k] = new_data[k / channels];
    }

    return *this;
}

double *Image::getGaussianKernel(int radius)
{

    const double EulerConstant = std::exp(1.0);

    double sigma = std::max((double)radius / 2, 1.0);

    double sum = 0;

    int ker_w = (2 * radius) + 1;

    double kernel[ker_w][ker_w];

    for (int i = -radius; i < radius; ++i)
    {
        for (int j = -radius; j < radius; ++j)
        {
            double exponentNumerator = i * i + j * j;

            double exponentDenominator = (2 * sigma * sigma);

            double eExpression = pow(EulerConstant, exponentNumerator / exponentDenominator);
            double kernelValue = (eExpression / (2 * M_PI * sigma * sigma));

            // We add radius to the indices to prevent out of bound issues because x and y can be negative
            kernel[i + radius][j + radius] = kernelValue;
            sum += kernelValue;
        }
    }

    double *flat_kernel = new double[ker_w * ker_w];

    for (size_t i = 0; i < ker_w; i++)
    {
        for (size_t j = 0; j < ker_w; j++)
        {
            flat_kernel[i * ker_w + j] = kernel[i][j] / sum;
        }
    }

    for (size_t i = 0; i < ker_w * ker_w; i++)
    {
        printf("%f, \n", flat_kernel[i]);
    }

    return flat_kernel;
}
