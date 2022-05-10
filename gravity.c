#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>

unsigned long NSTEPS, WRITE_INTERVAL, NBODIES;
double G, DT, DAMPING, SOFTENING;
int USE_THREADS;
int SQRTNT;

struct Body {
    double* x;
    double* y;
    double* z;
    double* vx;
    double* vy;
    double* vz;
};

struct tData {
    int chunk_i;
    int chunk_j;
    unsigned long chunk_size;
    struct Body** bodies;
};

PyObject* init_space;

void setSpace(struct Body *bodies[], PyArrayObject *space_arr){
    for (unsigned long i=0;i<NBODIES;i++){
        bodies[i] = (struct Body*)malloc(sizeof(struct Body));
        bodies[i]->x = (double*)PyArray_GETPTR3(space_arr,i,0,0);
        bodies[i]->y = (double*)PyArray_GETPTR3(space_arr,i,1,0);
        bodies[i]->z = (double*)PyArray_GETPTR3(space_arr,i,2,0);
        bodies[i]->vx = (double*)PyArray_GETPTR3(space_arr,i,0,1);
        bodies[i]->vy = (double*)PyArray_GETPTR3(space_arr,i,1,1);
        bodies[i]->vz = (double*)PyArray_GETPTR3(space_arr,i,2,1);
    }
}

void writeSpace(struct Body *bodies[]){
    FILE *f = fopen("result.data", "a");
    for (unsigned long i=0;i<NBODIES;i++) {
        fprintf(f, "%f %f %f\n", *bodies[i]->x, *bodies[i]->y, *bodies[i]->z);
    }
    fwrite("\n", sizeof(char), 1, f);
    fclose(f);
}

void forwardGravitation(struct Body *a, struct Body *b) {
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

void forwardVelocity(struct Body *bodies[]) {
    for (unsigned long i = 0; i < NBODIES; i++) {
        for (unsigned long j = 0; j < i; j++) {
            forwardGravitation(bodies[i], bodies[j]);
        }
    }
}

void forwardTriangleStep(unsigned long i, unsigned long j, struct Body *bodies[]) {
    /* Takes i,j in [0, NBODIES/2], perform two steps of the triangular matrix */
    unsigned long ii = NBODIES / 2 + i;
    unsigned long jj = NBODIES / 2 + j;
    forwardGravitation(bodies[ii], bodies[j]);
    if (i < j) {
        forwardGravitation(bodies[i], bodies[j]);
    } else {
        forwardGravitation(bodies[ii], bodies[jj]);
    }
}

void *forwardTriangleChunk(void* vtd) {
    struct tData* td=(struct tData*) vtd;
    for (unsigned long i = td->chunk_i * td->chunk_size; i < (td->chunk_i + 1) * td->chunk_size; i++) {
        for (unsigned long j = td->chunk_j * td->chunk_size; j < (td->chunk_j + 1) * td->chunk_size; j++) {
            forwardTriangleStep(i, j, td->bodies);
        }
    }
    return NULL;
}


void forwardVelocityThreads(struct Body *bodies[]) {
    unsigned long chunk_size = NBODIES / 2 / SQRTNT;
    pthread_t pth[SQRTNT * SQRTNT];
    for (int chunk_i = 0; chunk_i < SQRTNT; chunk_i++){
        for (int chunk_j = 0; chunk_j < SQRTNT; chunk_j++) {
            struct tData td;
            td.chunk_i = chunk_i;
            td.chunk_j = chunk_j;
            td.chunk_size = chunk_size;
            td.bodies = bodies;
            forwardTriangleChunk(&td);
            if (pthread_create(&pth[chunk_i * SQRTNT + chunk_j], NULL, forwardTriangleChunk, &td)) {
                printf("Alert! Error creating thread! Exiting Now!");
                exit(-1);
            }
        }
    }
    for (int chunk_i = 0; chunk_i < SQRTNT; chunk_i++){
        for (int chunk_j = 0; chunk_j < SQRTNT; chunk_j++) {
            pthread_join(pth[chunk_i * SQRTNT + chunk_j], NULL);
        }
    }
}

void forwardPosition(struct Body *bodies[]) {
    for (unsigned long i = 0; i < NBODIES; i++) {
        *bodies[i]->x += DT * (*bodies[i]->vx);
        *bodies[i]->y += DT * (*bodies[i]->vy);
        *bodies[i]->z += DT * (*bodies[i]->vz);
    };
}

static PyObject * run(PyObject* Py_UNUSED(self), PyObject* args) {
    if (!PyArg_ParseTuple(args, "Olddddlii", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL, &USE_THREADS, &SQRTNT)) {
        return NULL;
    }
    printf("Use thread yes/no: %d\n", USE_THREADS);
    PyArrayObject *space_arr = (PyArrayObject *) PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    NBODIES = PyArray_DIMS(space_arr)[0];
    struct Body *bodies[NBODIES];
    setSpace(bodies, space_arr);
    FILE *f = fopen("result.data", "w");
    fclose(f);
    double total_time = 0;
    for (unsigned long i = 0; i < NSTEPS; i++) {
        clock_t t = clock();
        if (USE_THREADS == 0) {
            forwardVelocity(bodies);
        } else {
            forwardVelocityThreads(bodies);
        }
        forwardPosition(bodies);
        total_time += (double)(clock() - t);
        if (i % WRITE_INTERVAL == 0) {
            printf("%ld\r", i);
            writeSpace(bodies);
        }
    }
    printf("\nC: %f seconds to execute\n", total_time / CLOCKS_PER_SEC);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity = { PyModuleDef_HEAD_INIT, .m_name = "_gravity", .m_methods = module_methods };

PyMODINIT_FUNC PyInit__gravity(void) { Py_Initialize(); import_array(); return PyModule_Create(&gravity); }
