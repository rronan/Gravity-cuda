#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>

struct Body {
    double* x;
    double* y;
    double* z;
    double* vx;
    double* vy;
    double* vz;
};

PyObject* init_space;

struct Config {
    unsigned long* nbodies;
    double* G;
    double* DT;
    double* DAMPING;
    double* SOFTENING;
} config;

typedef struct ThreadData {
    int chunk_i;
    int chunk_j;
    unsigned long chunk_size;
    struct Body* bodies[];
    struct Config* config;
} ThreadData;

void setSpace(struct Body *bodies[], PyArrayObject *space_arr, unsigned long nbodies){
    for (unsigned long i=0;i<nbodies;i++){
        bodies[i] = (struct Body*)malloc(sizeof(struct Body));
        bodies[i]->x = (double*)PyArray_GETPTR3(space_arr,i,0,0);
        bodies[i]->y = (double*)PyArray_GETPTR3(space_arr,i,1,0);
        bodies[i]->z = (double*)PyArray_GETPTR3(space_arr,i,2,0);
        bodies[i]->vx = (double*)PyArray_GETPTR3(space_arr,i,0,1);
        bodies[i]->vy = (double*)PyArray_GETPTR3(space_arr,i,1,1);
        bodies[i]->vz = (double*)PyArray_GETPTR3(space_arr,i,2,1);
    }
}

void parseArgs(unsigned long nbodies, double G, double DT, double DAMPING, double SOFTENING, int SQRTNT){
    unsigned long NSTEPS, WRITE_INTERVAL;
    double G, DT, DAMPING, SOFTENING;
    int SQRTNT;
    if (!PyArg_ParseTuple(args, "Olddddli", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL, &SQRTNT)) {
        return NULL;
    }
    config->G = double* G;
    config->DT = double* DT;
    config->DAMPING = double* DAMPING;
    config->SOFTENING = double* SOFTENING;
}

void writeSpace(struct Body *bodies[], unsigned long nbodies){
    FILE *f = fopen("result.data", "a");
    for (unsigned long i=0;i<nbodies;i++) {
        fprintf(f, "%f %f %f\n", *bodies[i]->x, *bodies[i]->y, *bodies[i]->z);
    }
    fwrite("\n", sizeof(char), 1, f);
    fclose(f);
}

void forwardGravitation(struct Body *a, struct Body *b, double G, double DT, double DAMPING, double SOFTENING) {
    double px = pow(*a->x - *b->x, 2);
    double py = pow(*a->y - *b->y, 2);
    double pz = pow(*a->z - *b->z, 2);
    double r = pow(px + py + pz + SOFTENING, .5);
    double h = DT * G / (r * r * r);
    *a->vx = (*a->vx + h * (*b->x - *a->x)) * DAMPING;
    *a->vy = (*a->vy + h * (*b->y - *a->y)) * DAMPING;
    *a->vz = (*a->vz + h * (*b->z - *a->z)) * DAMPING;
    *b->vx = (*b->vx + h * (*a->x - *b->x)) * DAMPING;
    *b->vy = (*b->vy + h * (*a->y - *b->y)) * DAMPING;
    *b->vz = (*b->vz + h * (*a->z - *b->z)) * DAMPING;
}

void forwardVelocity(struct Body *bodies[], struct Config* config) {
    for (unsigned long i = 0; i < config->nbodies; i++) {
        for (unsigned long j = 0; j < i; j++) {
            forwardGravitation(bodies[i], bodies[j], config);
        }
    }
}

void forwardTriangleStep(unsigned long i, unsigned long j, struct Body *bodies[], struct Config* config) {
    /* Takes i,j in [0, nbodies/2], perform two steps of the triangular matrix */
    unsigned long ii = nbodies / 2 + i;
    unsigned long jj = nbodies / 2 + j;
    forwardGravitation(bodies[ii], bodies[j], G, DT, DAMPING, SOFTENING);
    if (i < j) {
        forwardGravitation(bodies[i], bodies[j], G, DT, DAMPING, SOFTENING);
    } else {
        forwardGravitation(bodies[ii], bodies[jj], G, DT, DAMPING, SOFTENING);
    }
}
void forwardTriangleChunk(struct ThreadData* thread_data) {
    for (unsigned long i = chunk_i * chunk_size; i < (chunk_i + 1) * chunk_size; i++) {
        for (unsigned long j = chunk_j * chunk_size; j < (chunk_j + 1) * chunk_size; j++) {
            forwardTriangleStep(i, j, bodies, nbodies, G, DT, DAMPING, SOFTENING);
        }
    }
}

void forwardVelocityThreads(struct Body *bodies[], struct Config *config, int SQRTNT) {
    unsigned long chunk_size = nbodies / 2 / SQRTNT;
    pthread_t pth[SQRTNT * SQRTNT];
    for (int chunk_i = 0; chunk_i < SQRTNT; chunk_i++){
        for (int chunk_j = 0; chunk_j < SQRTNT; chunk_j++) {
            ThreadData* thread_data = malloc(sizeof *args);
            thread_data->chunk_i = &chunk_i;
            thread_data->chunk_j = &chunk_j;
            thread_data->chunk_size = &chunk_size;
            thread_data->bodies = bodies;
            thread_data->config = config;
            if(pthread_create(&pth[i * SQRTNT + j], NULL, forwardTriangleChunk, thread_data)) {
                free(thread_data);
                //goto error_handler;
            }
            forwardTriangleChunk(chunk_i, chunk_j, chunk_size, bodies, config);
        }
    }
}

void forwardPosition(struct Body *bodies[], unsigned long nbodies, double DT) {
    for (unsigned long i = 0; i < nbodies; i++) {
        *bodies[i]->x += DT * (*bodies[i]->vx);
        *bodies[i]->y += DT * (*bodies[i]->vy);
        *bodies[i]->z += DT * (*bodies[i]->vz);
    };
}

static PyObject * run(PyObject* Py_UNUSED(self), PyObject* args) {
    parseArgs(args)
    PyArrayObject *space_arr = (PyArrayObject *) PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    config->nbodies = PyArray_DIMS(space_arr)[0];
    struct Body *bodies[nbodies];
    setSpace(bodies, space_arr, nbodies);
    FILE *f = fopen("result.data", "w");
    fclose(f);
    double total_time = 0;
    for (unsigned long i = 0; i < NSTEPS; i++) {
        clock_t t = clock();
        forwardVelocityThreads(bodies, config, SQRTNT);
        forwardPosition(bodies, config->nbodies, config->DT);
        total_time += (double)(clock() - t);
        if (i % WRITE_INTERVAL == 0) {
            printf("%ld\r", i + 1);
            writeSpace(bodies, nbodies);
        }
    }
    printf("\nC: %f seconds to execute\n", total_time / CLOCKS_PER_SEC);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity = { PyModuleDef_HEAD_INIT, .m_name = "_gravity", .m_methods = module_methods };

PyMODINIT_FUNC PyInit__gravity(void) { Py_Initialize(); import_array(); return PyModule_Create(&gravity); }
