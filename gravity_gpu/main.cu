#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>
#include <time.h>

int WRITE_INTERVAL, NSTEPS, N_lt, N;
__device__ int d_N;
__device__ float G, DT, DAMPING, SOFTENING;

PyObject* init_space;

void setSpace(float* p, float* v, PyArrayObject *space_arr) {
    for (int i=0;i<N;i++) {
        p[i] = *(float*)PyArray_GETPTR3(space_arr,i,0,0);
        p[i + N] = *(float*)PyArray_GETPTR3(space_arr,i,1,0);
        p[i + 2 * N] = *(float*)PyArray_GETPTR3(space_arr,i,2,0);
        v[i] = *(float*)PyArray_GETPTR3(space_arr,i,0,1);
        v[i + N] = *(float*)PyArray_GETPTR3(space_arr,i,1,1);
        v[i + 2 * N] = *(float*)PyArray_GETPTR3(space_arr,i,2,1);
    }
}

void writeSpace(float* p){
    FILE *file = fopen("result.data", "a");
    for (unsigned long i=0;i<N;i++) {
        fprintf(file, "%f %f %f\n", p[i], p[i + N], p[i + N * 2]);
    }
    fwrite("\n", sizeof(char), 1, file);
    fclose(file);
}

__global__
void forwardGravitation(float *p, float *f) {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = (int) ((1 + sqrt(1 + 8 * (float) k)) / 2);
    int j = k - i * (i - 1) / 2;
    float dx = pow(p[i] - p[j], 2);
    float dy = pow(p[i + d_N] - p[j + d_N], 2);
    float dz = pow(p[i + 2 * d_N] - p[j + 2 * d_N], 2);
    float r = pow(dx + dy + dz + SOFTENING, .5);
    float h = G / (r * r * r);
    f[k] = h * (p[j] - p[i]);
    f[k + d_N] = h * (p[j + d_N] - p[i + d_N]);
    f[k + 2 * d_N] = h * (p[j + 2 * d_N] - p[i + 2 * d_N]);
}

__global__
void sumAcceleration(float *v, float *f) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < d_N; i++) {
        if (j < i) {
            int k = i * (i - 1) / 2 + j;
            v[i] += DT * f[k];
            v[i + d_N] += DT * f[k + d_N];
            v[i + d_N * 2] += DT * f[k + d_N * 2];
        } else if (j > i) {
            int k = j * (j - 1) / 2 + i;
            v[i] -= DT * f[k];
            v[i + d_N] -= DT * f[k + d_N];
            v[i + d_N * 2] -= DT * f[k + d_N * 2];
        }
    }
}

__global__
void forwardPosition(float *p, float *v) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    p[n] += DT * v[n];
    p[n + d_N] += DT * v[n + d_N];
    p[n + 2 * d_N] += DT * v[n + 2 * d_N];
}

static PyObject * run(PyObject* Py_UNUSED(self), PyObject* args) {
    if (!PyArg_ParseTuple(args, "Olddddl", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL)) {
        return NULL;
    }
    PyArrayObject *space_arr = (PyArrayObject *) PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    N = PyArray_DIMS(space_arr)[0];
    cudaMemcpy(&d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    N_lt = N * (N - 1) / 2;
    float *p;
    float *v;
    p = (float*)malloc(3 * sizeof(float) * N);
    v = (float*)malloc(3 * sizeof(float) * N);
    setSpace(p, v, space_arr);
    float *d_p, *d_v, *f;
    cudaMalloc(&d_p, 3 * sizeof(float) * N);
    cudaMalloc(&d_v, 3 * sizeof(float) * N);
    cudaMalloc(&f, 3 * sizeof(float) * N_lt);
    cudaMemcpy(d_p, p, 3 * sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, 3 * sizeof(float) * N, cudaMemcpyHostToDevice);
    FILE *file = fopen("result.data", "w");
    fclose(file);
    time_t t0 = time(NULL);
    for (unsigned long i = 0; i < NSTEPS; i++) {
        forwardGravitation<<<(3*N_lt+255)/256, 256>>>(d_p, f);
        sumAcceleration<<<(3*N+255)/256, 256>>>(d_v, f);
        forwardPosition<<<(3*N+255)/256, 256>>>(d_p, d_v);
        if (i % WRITE_INTERVAL == 0) {
            printf("%ld\r", i);
            cudaMemcpy(p, d_p, 3 * sizeof(float) * N, cudaMemcpyDeviceToHost);
            writeSpace(p);
        }
    }
    printf("Total time: %f\n", (float) (time(NULL) - t0));
    free(p);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(f);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity_gpu = { PyModuleDef_HEAD_INIT, .m_name = "gravity_gpu", .m_methods = module_methods };

PyMODINIT_FUNC PyInit_gravity_gpu(void) { Py_Initialize(); import_array(); return PyModule_Create(&gravity_gpu); }
