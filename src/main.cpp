
#include "Image.h"
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <string.h>

#define MAX(a, b) a < b ? b : a

void make_random_complex_array(uint32_t len, std::complex<double> *z)
{
    for (uint32_t i = 0; i < len; ++i)
    {
        z[i] = std::complex<double>(rand() % 100, rand() % 100);
    }
}

void print_complex_array(uint32_t len, std::complex<double> z[])
{
    for (uint32_t i = 0; i < len; ++i)
    {
        printf("%f+%fi, \n", z[i].real(), z[i].imag());
    }
    printf("\n");
}

int process(Image image, int len, std::string type)
{
    double *big_kernel = new double[len];

    for (uint16_t i = 0; i < len; i++)
    {
        big_kernel[i] = 1 / (double)len;
    }

    Image standart = image;
    Image fftway = image;

    auto std_conv_start = std::chrono::system_clock::now();

    standart.std_convolve((uint8_t)sqrt(len), (uint8_t)sqrt(len), big_kernel, 0, 0);

    auto std_conv_end = std::chrono::system_clock::now();

    auto fd_conv_start = std::chrono::system_clock::now();

    fftway.fd_convolve((uint8_t)sqrt(len), (uint8_t)sqrt(len), big_kernel, 0, 0);

    auto fd_conv_end = std::chrono::system_clock::now();

    printf("Fourier method took %f seconds\n", std::chrono::duration_cast<std::chrono::nanoseconds>(fd_conv_end - fd_conv_start).count() * pow(10, -9));
    printf("Standart method took %f seconds\n", std::chrono::duration_cast<std::chrono::nanoseconds>(std_conv_end - std_conv_start).count() * pow(10, -9));

    std::string length_string = "_" + std::to_string((uint32_t)sqrt(len)) + "_";
    std::string std_method_filename = "data/" + type + length_string + "x" + length_string + "std_method.jpg";
    std::string fd_method_filename = "data/" + type + length_string + "x" + length_string + "fd_method.jpg";

    char *standart_chars = strdup(std_method_filename.c_str());
    char *fd_chars = strdup(fd_method_filename.c_str());

    standart.write(standart_chars);
    fftway.write(fd_chars);

    free(standart_chars);
    free(fd_chars);

    delete[] big_kernel;
}

int main(int argc, char **argv)
{
    Image big_img("test.jpg");

    Image small_img("download.jpg");

    printf("-----------------------------------\n");
    printf("Test box blur for small image\n");
    printf("-----------------------------------\n");
    for (int i = 2; i <= 30; i++)
    {
        printf("-----------------------------------\n");
        printf("for matrix %d x %d\n\n", i, i);
        process(small_img, i * i, "small_image");
        printf("\n\n");
    }

    printf("-----------------------------------\n");
    printf("Test box blur for big image\n");
    printf("-----------------------------------\n");
    for (int i = 2; i <= 30; i++)
    {
        printf("-----------------------------------\n");
        printf("for matrix %d x %d\n\n", i, i);
        process(big_img, i * i, "big_image");
        printf("\n\n");
    }

    return 0;
}