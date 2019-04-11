all: radix_sort

radix_sort: src/radixsort.cu
	nvcc src/radixsort.cu -o radix_sort