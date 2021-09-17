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
    int* p = (int*) rram_malloc(2 * sizeof(int));
    // try more addresses on larger scale
    p[0] = p[1] + 23;
    cout << p[0] << endl;
    int* p2 = (int*) rram_malloc(10 * sizeof(int));
    p2[0] = p[0] + 24;
    cout << p2[0] << endl;

    int* nopin = (int*) malloc(0x20);
    nopin[0] = 0x98;
    free(nopin);

    return 0;
}
