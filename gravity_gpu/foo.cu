#include <stdio.h>

__global__
void forwardGravitation(float *p, float *f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = (int) ((1 + sqrt(1 + 8 * (float) k)) / 2);
    int j = k - i * (i - 1) / 2;
    float dx = pow(p[i] - p[j], 2);
    float r = sqrt(dx + 0.01);
    float h = 10 / (r * r * r);
    f[k] = h * (p[j] - p[i]);
}

__global__
void sumAcceleration(float *v, float *f) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = 2;
    for (int j = 0; j < N; j++) {
        if (j < i) {
            int k = i * (i - 1) / 2 + j;
            v[i] += 1e-4 * f[k];
        } else if (j > i) {
            int k = j * (j - 1) / 2 + i;
            v[i] -= 1e-4 * f[k];
        }
    }
}



__global__
void forwardPosition(float *p, float *v) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    p[n] += 1e-4 * v[n];
}

int main() {
    printf("Start gravity_gpu!\n"); fflush(stdout);
    int NSTEPS = 400;
    int N = 2;
    int N_lt = 1;
    /* Init CPU arrays */
    float *p, *v, *f;
    p = (float*)malloc(sizeof(float) * N);
    v = (float*)malloc(sizeof(float) * N);
    f = (float*)malloc(sizeof(float) * N_lt); // DEBUG
    p[0] = 0.0;
    p[1] = 1.0;
    v[0] = 0.0;
    v[1] = 0.0;
    f[0] = 0.0;
    f[1] = 0.0;
    /* Init CUDA arrays */
    float *d_p, *d_v, *d_f;
    cudaMalloc(&d_p, sizeof(float) * N);
    cudaMalloc(&d_v, sizeof(float) * N);
    cudaMalloc(&d_f, sizeof(float) * N_lt);
    cudaMemcpy(d_p, p, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, sizeof(float) * N_lt, cudaMemcpyHostToDevice);
    for (int _ = 0; _ < NSTEPS; _++) {
        /* p */
        cudaMemcpy(p, d_p, sizeof(float) * N, cudaMemcpyDeviceToHost);
        printf("%f   %f\n", p[0], p[1]); fflush(stdout);
        /* /1* v *1/ */
        /* cudaMemcpy(v, d_v, sizeof(float) * N, cudaMemcpyDeviceToHost); */
        /* for (int j=0; j<N; j++) printf("v[%d] %f\n", j, v[j]); fflush(stdout); */
        /* /1* f *1/ */
        /* cudaMemcpy(f, d_f, sizeof(float) * N_lt, cudaMemcpyDeviceToHost); */
        /* for (int j=0; j<N_lt; j++) printf("f[%d] %f\n", j, f[j]); fflush(stdout); */

        forwardGravitation<<<1, N_lt>>>(d_p, d_f);
        sumAcceleration<<<1, N>>>(d_v, d_f);
        forwardPosition<<<1, N>>>(d_p, d_v);

        /* printf("after\n"); fflush(stdout); */

        /* /1* p *1/ */
        /* cudaMemcpy(p, d_p, sizeof(float) * N, cudaMemcpyDeviceToHost); */
        /* for (int j=0; j<N; j++) printf("p[%d] %f\n", j, p[j]); fflush(stdout); */
        /* /1* v *1/ */
        /* cudaMemcpy(v, d_v, sizeof(float) * N, cudaMemcpyDeviceToHost); */
        /* for (int j=0; j<N; j++) printf("v[%d] %f\n", j, v[j]); fflush(stdout); */
        /* /1* f *1/ */
        /* cudaMemcpy(f, d_f, sizeof(float) * N_lt, cudaMemcpyDeviceToHost); */
        /* for (int j=0; j<N_lt; j++) printf("f[%d] %f\n", j, f[j]); fflush(stdout); */

        /* printf("++++++++++++++++++++++++++++++\n"); fflush(stdout); */
    }
    return 0;
}
