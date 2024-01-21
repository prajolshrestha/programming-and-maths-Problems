#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 1000

int A[N][N];
int B[N][N];
int C[N][N];

typedef struct {
    int row;
    int col;
} Args;

void *thread_worker(void *arg) {
    Args *args = (Args*)arg;
    int row = args->row;
    int col = args->col;

    int sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[row][k] * B[k][col];
    }
    C[row][col] = sum;

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[N][N];
    Args args[N][N];

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    // Create threads and assign tasks
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            args[i][j].row = i;
            args[i][j].col = j;
            pthread_create(&threads[i][j], NULL, thread_worker, &args[i][j]);
        }
    }

    // Wait for threads to complete
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            pthread_join(threads[i][j], NULL);
        }
    }

    // Print matrix C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
