#include <Python.h>

int NTHREADS = 4;

struct ThreadData {
    int i;
    struct Body* bodies[];
};

void forwardTriangleChunk(struct ThreadData* td) {
    return NULL;
}

void forwardVelocityThreads(struct Body *bodies[]) {
    pthread_t pth[NTHREADS];
    for (int i = 0; i < NTHREADS; i++){
        struct ThreadData td;
        td->i = i;
        // HELP: the following does not work:
        td->bodies = bodies; // error: array type 'struct Body *[]' is not assignable
        pthread_create(&pth[i], NULL, forwardTriangleChunk, td)
    }
}

static void main():
    // nbodies is defined at runtime
    int nbodies = 10;
    struct Body *bodies[nbodies];
    forwardVelocityThreads(bodies)
