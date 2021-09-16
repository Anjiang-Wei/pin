#include <iostream>
#include <stdlib.h>
using namespace std;

void rram_free(void* p)
{
    return free(p);
}

void* rram_malloc(size_t size)
{
    cout << "rram malloc" << endl;
    void* p1 = malloc(size);
    return p1;
}

int main()
{
    int* p = (int*) rram_malloc(0x10);
    p[0] = 0x89;
    // try more addresses on larger scale
    rram_free(p);

    int* nopin = (int*) malloc(0x20);
    nopin[0] = 0x98;
    free(nopin);
    return 0;
}
