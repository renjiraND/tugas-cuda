#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define WSIZE 32
#define LOOPS 100
#define UPPER_BIT 10
#define LOWER_BIT 0


__device__ unsigned int ddata[WSIZE];
__device__ int ddata_s[WSIZE];

__device__ int getMax(int arr[], int n)
{
	int mx = arr[0];
	for (int i = 1; i < n; i++)
		if (arr[i] > mx)
			mx = arr[i];
	return mx;
}

__device__ void countSort(int arr[], int n, int exp)
{
	int output[n]; // Output array
	int i, count[10] = { 0 };

	// simpan banyaknya kemunculan digit 
	for (i = 0; i < n; i++) {
		count[(arr[i] / exp) % 10]++;
	}

	for (i = 1; i < 10; i++) {
		count[i] += count[i - 1];
	}

	// output array: terurut berdasarkan current digit
	for (i = n - 1; i >= 0; i--) {
		output[count[(arr[i] / exp) % 10] - 1] = arr[i];
		count[(arr[i] / exp) % 10]--;
	}

	// Copy output[] ke arr[]
	for (i = 0; i < n; i++)
		arr[i] = output[i];
}

__device__ void radixsort(int arr[], int n) {
	// berdasarkan digit
	int m = getMax(arr, n);

	// exp adalah 10^i
	// i = current digit
	for (int exp = 1; m / exp > 0; exp *= 10) {
		countSort(arr, n, exp);
	}
}

__global__ void serialRadix() {
	radixsort(ddata_s, WSIZE);
}

__global__ void parallelRadix() {
	// shared memory
	__shared__ volatile unsigned int sdata[WSIZE * 2];

	// load dari global ke shared
	sdata[threadIdx.x] = ddata[threadIdx.x];

	unsigned int bitmask = 1 << LOWER_BIT;
	unsigned int offset = 0;
	unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
	unsigned int pos;

	
	for (int i = LOWER_BIT; i <= UPPER_BIT; i++)
	{
		unsigned int mydata = sdata[((WSIZE - 1) - threadIdx.x) + offset];
		unsigned int mybit = mydata&bitmask;

		unsigned int satu = __ballot(mybit);
		unsigned int nol = ~satu;
		offset ^= WSIZE;

		if (!mybit) {
			pos = __popc(nol & thrmask);
		} else  {
			pos = __popc(nol) + __popc(satu & thrmask);
		}

		sdata[pos - 1 + offset] = mydata;
		bitmask <<= 1;
	}
	ddata[threadIdx.x] = sdata[threadIdx.x + offset];
}

void rng(unsigned* arr, int n) {
    int seed = 13516014; // NIM Renjira
    srand(seed);
    for(long i = 0; i < n; i++) {
        arr[i] = (int)rand();
    }
}

int main() {
	/* Parallel */
	int arr_size = (WSIZE * LOOPS);
	unsigned int hdata[arr_size];
	float totalTime = 0;

	// isi array dengan random element
	rng(hdata, arr_size);
	for (int j = 0; j < LOOPS; j++) {
		srand(time(NULL));

		// Copy data host ke device
		cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(unsigned int));

		// waktu awal
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		parallelRadix <<< 1, WSIZE >>>();
		// synchronous kernel
		cudaDeviceSynchronize();
		// waktu akhir
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		// durasi
		auto duration = duration_cast<milliseconds>(t2 - t1).count();
		totalTime += (float)duration * 1000.00;

		// Copy data from device to host
		cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(unsigned int));
	}

	printf("\nParallel :\n");
	printf("Array size = %d\n", arr_size);
	printf("Time elapsed = %fmicroseconds\n", totalTime);


	/* Serial */
	unsigned int hdata_s[arr_size];
	totalTime = 0;

	// isi array dengan random element
	rng(hdata, arr_size);
	for (int j = 0; j < LOOPS; j++) {
		srand(time(NULL));
		
		// Copy data host ke device
		cudaMemcpyToSymbol(ddata_s, hdata_s, WSIZE * sizeof(unsigned int));

		// waktu awal
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		serialRadix <<< 1, 1 >>>();
		// synchronous kernel
		cudaDeviceSynchronize();
		// waktu akhir
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		// durasi
		auto duration = duration_cast<milliseconds>(t2 - t1).count();
		totalTime += (float)duration * 1000.00;

		// Copy data from device to host
		cudaMemcpyFromSymbol(hdata_s, ddata_s, WSIZE * sizeof(unsigned int));
	}

	printf("\nSerial :\n");
	printf("Array size = %d\n", arr_size);
	printf("Time elapsed = %fmicroseconds\n\n", totalTime);

	return 0;
}