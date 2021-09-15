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
    void* p = rram_malloc(16);
    rram_free(p);
    return 0;
}
