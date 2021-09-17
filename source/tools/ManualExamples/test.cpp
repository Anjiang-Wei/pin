#include <iostream>
#include <stdlib.h>
using namespace std;

void* rram_malloc(size_t size)
{
    cout << "rram malloc" << endl;
    void* p1 = malloc(size);
    return p1;
}

void compute(char* ptr, int test_size)
{
    for (int i = 0; i < test_size; i++)
    {
        ptr[i] = i;
    }
    for (int i = test_size - 1; i >= 1; i--)
    {
        ptr[0] += ptr[i];
    }
}

void compute(int* ptr, int test_size)
{
    for (int i = 0; i < test_size; i++)
    {
        ptr[i] = i;
    }
    for (int i = test_size - 1; i >= 1; i--)
    {
        ptr[0] += ptr[i];
    }
}

void compute(float* ptr, int test_size)
{
    for (int i = 0; i < test_size; i++)
    {
        ptr[i] = i;
    }
    for (int i = test_size - 1; i >= 1; i--)
    {
        ptr[0] += ptr[i];
    }
}

void compute(double* ptr, int test_size)
{
    for (int i = 0; i < test_size; i++)
    {
        ptr[i] = i;
    }
    for (int i = test_size - 1; i >= 1; i--)
    {
        ptr[0] += ptr[i];
    }
}


int main()
{
    int test_size = 101;
    
    char* p0 = (char*) rram_malloc(test_size * sizeof(char));
    compute(p0, 10);
    cout << p0[0] << endl;

    int* p1 = (int*) rram_malloc(test_size * sizeof(int));
    compute(p1, test_size);
    cout << p1[0] << endl;
    
    float* p2 = (float*) rram_malloc(test_size * sizeof(float));
    compute(p2, test_size);
    cout << p2[0] << endl;

    double* p3 = (double*) rram_malloc(test_size * sizeof(double));
    compute(p3, test_size);
    cout << p3[0] << endl;

    int* nopin = (int*) malloc(test_size * sizeof(int));
    cout << "nothing should happen" << endl;
    nopin[0] = 0x98;
    free(nopin);
    free(p0);
    free(p1);
    free(p2);
    free(p3);

    return 0;
}
