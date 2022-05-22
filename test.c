#include <stdlib.h>
#include <stdio.h>
/* #include <omp.h> */
#include <libiomp/omp.h>
#include <time.h>

void process(long i, long j, long nbodies, double *arr){
    long k = i * nbodies + j;
    printf("%ld\n", k);
    arr[k] = (double)(k);
}

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    long nbodies = (long)strtol(argv[1], NULL, 10);
    printf("Length: %ld\n", nbodies); fflush(stdout);
    double *arr = (double*)malloc(sizeof(double) * nbodies);
    clock_t t = clock();
    #pragma omp parallel for
    for (long i=0; i<nbodies; i++){
        printf("%ld\n", i);
        arr[i] = (double)(i);
        /* process(i, j, nbodies, arr); */
    }
    double total_time = (double)(clock() - t);
    printf("%f seconds to execute\n", total_time / CLOCKS_PER_SEC);
    return 0;
}
