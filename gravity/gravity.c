#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL NP_ARRAY_API
#include <numpy/arrayobject.h>

unsigned long NSTEPS, WRITE_INTERVAL, NBODIES;
double G, DT, DAMPING, SOFTENING;
int USE_THREADS;
int NTHREADS;

unsigned long COUNTER=0;

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
    unsigned long chunk_size;
    struct Body** bodies;
    double*** dv;
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
    COUNTER += 1;
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


void forwardGravitationMemory(unsigned long i, unsigned long j, struct Body *bodies[], double ***dv) {
    COUNTER += 1;
    struct Body* a = bodies[i];
    struct Body* b = bodies[j];
    double px = pow(*a->x - *b->x, 2);
    double py = pow(*a->y - *b->y, 2);
    double pz = pow(*a->z - *b->z, 2);
    double r = pow(px + py + pz + SOFTENING, .5);
    double h = DT * G / (r * r * r);
    dv[i][j][0] = (*a->vx + h * (*b->x - *a->x)) * DAMPING;
    dv[i][j][1] = (*a->vy + h * (*b->y - *a->y)) * DAMPING;
    dv[i][j][2] = (*a->vz + h * (*b->z - *a->z)) * DAMPING;
    dv[j][i][0] = (*b->vx + h * (*a->x - *b->x)) * DAMPING;
    dv[j][i][1] = (*b->vy + h * (*a->y - *b->y)) * DAMPING;
    dv[j][i][2] = (*b->vz + h * (*a->z - *b->z)) * DAMPING;
}

void forwardTriangleStep(unsigned long i, unsigned long j, struct Body *bodies[], double ***dv) {
    /* Takes i,j in [0, NBODIES/2], perform two steps of the triangular matrix */
    unsigned long ii = NBODIES / 2 + i;
    unsigned long jj = NBODIES / 2 + j;
    forwardGravitationMemory(ii, j, bodies, dv);
    if (i < j) {
        forwardGravitationMemory(i, j, bodies, dv);
    } else {
        forwardGravitationMemory(ii, jj, bodies, dv);
    }
}

void *forwardTriangleChunk(void* vtd) {
    struct tData* td=(struct tData*) vtd;
    for (unsigned long i = 0; i < td->chunk_size; i++) {
        for (unsigned long j = 0; j < NBODIES / 2; j++) {
            forwardTriangleStep(td->chunk_i * td->chunk_size + i, j, td->bodies, td->dv);
        }
    }
    return NULL;
}

void *sumAccelerations(void* vtd) {
    struct tData* td=(struct tData*) vtd;
    for (unsigned long i = 0; i < td->chunk_size; i++) {
        for (unsigned long j = 0; j < NBODIES; j++) {
            *td->bodies[i]->vx += td->dv[i][j][0];
            *td->bodies[i]->vy += td->dv[i][j][1];
            *td->bodies[i]->vz += td->dv[i][j][2];
        }
    }
    return NULL;
}


void forwardVelocityThreads(struct Body *bodies[], double ***dv) {
    unsigned long chunk_size = NBODIES / 2 / NTHREADS;
    pthread_t pth[NTHREADS];
    struct tData td;
    td.bodies = bodies;
    td.dv = dv;
    for (int chunk_i = 0; chunk_i < NTHREADS; chunk_i++){
        td.chunk_i = chunk_i;
        td.chunk_size = chunk_size;
        if (pthread_create(&pth[chunk_i], NULL, forwardTriangleChunk, &td)) {
            printf("Alert! Error creating thread! Exiting Now!");
            exit(-1);
        }
    }
    for (int chunk_i = 0; chunk_i < NTHREADS; chunk_i++){
        pthread_join(pth[chunk_i], NULL);
    }
    for (int chunk_i = 0; chunk_i < NTHREADS; chunk_i++){
        struct tData td;
        td.chunk_i = chunk_i;
        td.chunk_size = chunk_size;
        if (pthread_create(&pth[chunk_i], NULL, sumAccelerations, &td)) {
            printf("Alert! Error creating thread! Exiting Now!");
            exit(-1);
        }
    }
    for (int chunk_i = 0; chunk_i < NTHREADS; chunk_i++){
        pthread_join(pth[chunk_i], NULL);
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
    if (!PyArg_ParseTuple(args, "Olddddlii", &init_space, &NSTEPS, &G, &DT, &DAMPING, &SOFTENING, &WRITE_INTERVAL, &USE_THREADS, &NTHREADS)) {
        return NULL;
    }
    printf("Use thread yes/no: %d\n", USE_THREADS);
    PyArrayObject *space_arr = (PyArrayObject *) PyArray_ContiguousFromObject(init_space, NPY_DOUBLE, 0, 0);
    NBODIES = PyArray_DIMS(space_arr)[0];
    struct Body *bodies[NBODIES];
    setSpace(bodies, space_arr);
    double **dv[NBODIES];
    for (unsigned long i = 0; i < NSTEPS; i++) {
        double *dv_[3];
        for (unsigned long j = 0; j < NSTEPS; i++) {
            double dv__[3] = {0, 0, 0};
            dv_[j] = dv__;
        }
        dv[i] = dv_;
    }
    FILE *f = fopen("result.data", "w");
    fclose(f);
    double total_time = 0;
    for (unsigned long i = 0; i < NSTEPS; i++) {
        clock_t t = clock();
        if (USE_THREADS == 0) {
            forwardVelocity(bodies);
        } else {
            forwardVelocityThreads(bodies, dv);
        }
        forwardPosition(bodies);
        total_time += (double)(clock() - t);
        if (i % WRITE_INTERVAL == 0) {
            printf("%ld\r", i);
            writeSpace(bodies);
        }
    }
    printf("\nC: %ld Number of forwardGravitation()\n", COUNTER);
    printf("C: %f seconds to execute\n", total_time / CLOCKS_PER_SEC);
    return Py_None;
}

static PyMethodDef module_methods[] = {{"run", run, METH_VARARGS, NULL}, {0, 0}};

static struct PyModuleDef gravity = { PyModuleDef_HEAD_INIT, .m_name = "_gravity", .m_methods = module_methods };

PyMODINIT_FUNC PyInit__gravity(void) { Py_Initialize(); import_array(); return PyModule_Create(&gravity); }
