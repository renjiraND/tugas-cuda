#include <stdio.h>
#include <math.h>
#include <thrust/scan.h>

// fill random number to array
void rng(int* arr, int n) {
    int seed = 13516014; // NIM Renjira
    srand(seed);
    for (long i = 0; i < n; i++) {
        arr[i] = (int) rand();
    }
}

// get max number in array
__host__ int getMax(int *arr, int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

__global__ void counting(int *arr, int *count, int n, int exp) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int digit = (arr[i] / exp) % 10;
        atomicAdd(&count[digit], 1);
    }
    __syncthreads();
}

__host__ void parallelCountSort(int *arr, int n, int exp) {
    int output[n];

    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof (int));
    cudaMemcpy(d_arr, arr, n * sizeof (int), cudaMemcpyHostToDevice);
    int count[10] = {0};
    int *d_count;
    cudaMalloc(&d_count, 10 * sizeof (int));
    cudaMemcpy(d_count, count, 10 * sizeof (int), cudaMemcpyHostToDevice);


    int grid_size = (n + 1023) / 1024;
    int thread_size = 1024;

    counting << <grid_size, thread_size>>>(d_arr, d_count, n, exp);

    cudaMemcpy(count, d_count, 10 * sizeof (int), cudaMemcpyDeviceToHost);

    thrust::inclusive_scan(count, count + 10, count);


    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// parallel radix sort
__host__ void parallelRadixSort(int *arr, int n) {
    int m = getMax(arr, n);

    for (int exp = 1; m / exp > 0; exp *= 10) {
        parallelCountSort(arr, n, exp);
    }
}

// print array
__host__ void print(int *arr, int n) {
    printf("=======Array element:\n");
    for (int i = 0; i < n; i++) {
        printf("%d\n", arr[i]);
    }
    printf("=====================\n");
}

__host__ void serialCountSort(int *arr, int n, int exp) {
    int output[n];
    int count[10] = {0};

    for (int i = 0; i < n; i++) {
        int d = (arr[i] / exp) % 10;
        count[d]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        int d = (arr[i] / exp) % 10;
        output[ count[d] - 1 ] = arr[i];
        count[d]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

__host__ void serialRadixSort(int *arr, int n) {
    int m = getMax(arr, n);

    for (int exp = 1; m / exp > 0; exp *= 10) {
        serialCountSort(arr, n, exp);
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, finish;
    float elapsedTime;
    if (argc < 2) {
        printf("usage: ./main N\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int *arr = (int *) malloc(n * sizeof (int));
    printf("Array size = %d\n", n);
    rng(arr, n);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    serialRadixSort(arr, n);

    cudaEventCreate(&finish);
    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);

    cudaEventElapsedTime(&elapsedTime, start, finish);
    printf("Elapsed time for serial radix sort:  %f\n", elapsedTime);

    rng(arr, n);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    parallelRadixSort(arr, n);

    cudaEventCreate(&finish);
    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);

    cudaEventElapsedTime(&elapsedTime, start, finish);
    printf("Elapsed time for parallel radix sort:  %f\n", elapsedTime);
}