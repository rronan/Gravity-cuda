#include <stdlib.h>
#include <stdio.h>

void process(long length, double *arr){
    for (long i=0; i<length; i++){
        arr[i] = (double)i;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    long length = (long)strtol(argv[1], NULL, 10);
    printf("Length: %ld\n", length); fflush(stdout);
    double *arr = (double*)malloc(sizeof(double) * length);
    for (long i=0; i<length; i++){
        arr[i] = (double)i;
    }
    printf("First success\n"); fflush(stdout);
    process(length, arr);
    printf("Second success\n"); fflush(stdout);
    return 0;
}
