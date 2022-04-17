#include <Python.h>
#include <structmember.h>
#include <stddef.h>
#include <math.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define NBODIES 50

const float DAMPING  = 1 - 1e-7;
const float SOFTENING = 10;
const double R = 200;
const double V = 100;
const double G = 1e5;
const double DT = 1e-4;

struct Body {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
};

struct Body *bodies[NBODIES];

void setSpace(struct Body *bodies[NBODIES]){
    srand (time(NULL));
    double avg_vx = 0;
    double avg_vy = 0;
    double avg_vz = 0;
    int i = 0;
    while (i < NBODIES) {
        double x = (rand() * 2 - 1) * R / (double)RAND_MAX;
        double y = (rand() * 2 - 1) * R / (double)RAND_MAX;
        double z = (rand() * 2 - 1) * R / (double)RAND_MAX;
        double vx = (rand() * 2 - 1) * V / (double)RAND_MAX;
        double vy = (rand() * 2 - 1) * V / (double)RAND_MAX;
        double vz = (rand() * 2 - 1) * V / (double)RAND_MAX;
        if ((x * x + y * y + z * z) <= R * R) {
            bodies[i] = (struct Body*)malloc(sizeof(struct Body));
            bodies[i]->x = x;
            bodies[i]->y = y;
            bodies[i]->z = z;
            bodies[i]->vx = vx;
            bodies[i]->vy = vy;
            bodies[i]->vz = vz;
            avg_vx += vx;
            avg_vy += vy;
            avg_vz += vz;
            i++;
        }
    }
    for (i = 0; i < NBODIES; ++i) {
        bodies[i]->vx -= avg_vx / NBODIES;
        bodies[i]->vy -= avg_vy / NBODIES;
        bodies[i]->vz -= avg_vz / NBODIES;
    }
}

void forwardGravitation(struct Body *a, struct Body *b) {
    double px = pow(a->x - b->x, 2);
    double py = pow(a->y - b->y, 2);
    double pz = pow(a->z - b->z, 2);
    double r = pow(px + py + pz + SOFTENING, .5);
    double f = G / (r * r);
    a->vx = (a->vx + DT * f * (b->x - a->x) / r) * DAMPING;
    a->vy = (a->vy + DT * f * (b->y - a->y) / r) * DAMPING;
    a->vz = (a->vz + DT * f * (b->z - a->z) / r) * DAMPING;
}

void forwardPhysics(struct Body *bodies[NBODIES]) {
    for (unsigned long i = 0; i < NBODIES; i++) {
        for (unsigned long j = 0; j < NBODIES; j++) {
            if (i != j) {
                forwardGravitation(bodies[i], bodies[j]);
            }
        }
        bodies[i]->x += DT * bodies[i]->vx;
        bodies[i]->y += DT * bodies[i]->vy;
        bodies[i]->z += DT * bodies[i]->vz;
    };
}

static PyObject * run(PyObject* Py_UNUSED(self), PyObject* args) {
    unsigned long NSTEPS;
    if (!PyArg_ParseTuple(args, "l", &NSTEPS))
        return NULL;
    setSpace(bodies);
    clock_t t;
    t = clock();
    for (unsigned long i = 0; i < NSTEPS; i++) {
        forwardPhysics(bodies);
        printf("%ld\n", i);
    }
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("C: %f seconds to execute", time_taken);
    return Py_None;
}

static PyMethodDef module_methods[] = {
    {"run", run, METH_VARARGS, NULL}, {0, 0}
};

static struct PyModuleDef gravity = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_gravity",
    .m_methods = module_methods
};

PyMODINIT_FUNC
PyInit__gravity(void) {
    Py_Initialize();
    return PyModule_Create(&gravity);
}
