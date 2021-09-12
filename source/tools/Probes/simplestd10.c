/*
 * Copyright 2002-2020 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

// This little application is used to test calling application functions.

#include <stdio.h>

extern int __stdcall StdBar10(int, int, int, int, int, int, int, int, int, int);

int main()
{
    int sum;

    sum = StdBar10(1, 2, 3, 4, 5, 6, 7, 8, 9, 0);

    printf(" main: sum=%d\n", sum);

    return 0;
}
