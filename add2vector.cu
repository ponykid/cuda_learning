#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


/*
   vector add methods
   1. one block with multiple threads
   2. multiple blocks, each block one thread
   3. multiple blocks with multiple threads
*/
//#define N 1024*10
#define N 10000000
#define THREADSPERBLOCK 256

__global__ void VecAdd(int *A, int *B, int *C){
	int tid = threadIdx.x;
	/*
	while(tid < N);{
		C[tid] = A[tid] + B[tid];
		tid = tid + THREADSPERBLOCK;
	}
	*/

	C[tid] = A[tid] + B[tid];
}

__global__ void Add(int *A, int *B, int *C){
	int tid = blockIdx.x;
	/*
	while(tid < N);{
		C[tid] = A[tid] + B[tid];
		tid = tid + THREADSPERBLOCK;
	}
	*/

	C[tid] = A[tid] + B[tid];
}

// use multiple blocks and multiple threads
__global__ void Add3(int *A, int *B, int *C){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x * blockDim.x + blockIdx.x;
	/*
	while(tid < N);{
		C[tid] = A[tid] + B[tid];
		tid = tid + THREADSPERBLOCK;
	}
	*/

	C[tid] = A[tid] + B[tid];
}

__global__ void add3(int *A, int *B, int *C, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x * blockDim.x + blockIdx.x;
	/*
	while(tid < N);{
		C[tid] = A[tid] + B[tid];
		tid = tid + THREADSPERBLOCK;
	}
	*/

	if (tid < N)
		C[tid] = A[tid] + B[tid];
}

int main( void ){
	int *a, *b, *c;
	int * dev_a, *dev_b, *dev_c;
	// allocate the memory on the CPU
	a = (int *)malloc( N * sizeof(int) );
	b = (int *)malloc( N * sizeof(int) );
	c = (int *)malloc( N * sizeof(int) );

	// allocate the memory on the GPU
	cudaMalloc( (void ** )&dev_a, N *sizeof(int) );
	cudaMalloc( (void ** )&dev_b, N *sizeof(int) );
	cudaMalloc( (void ** )&dev_c, N *sizeof(int) );

	// initial a, b, c
	srand( time(NULL) );
	for (int i=0; i<N; i++){
		a[i] = int(rand() % 256);
		b[i] = int(rand() % 256);
	}

	// cpoy the array's 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice );

	// Get start time event
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	// GPU kernel function
	//VecAdd<<<1, N>>>(dev_a, dev_b, dev_c);
	//VecAdd<<<1, THREADSPERBLOCK>>>(dev_a, dev_b, dev_c);
	//Add<<<N,1>>>(dev_a, dev_b, dev_c);
	//Add3<<<1024,1024>>>(dev_a, dev_b, dev_c);
	Add3<<<(N + THREADSPERBLOCK-1)/THREADSPERBLOCK, THREADSPERBLOCK>>>(dev_a, dev_b, dev_c);
	//Add3<<<(N)/THREADSPERBLOCK, THREADSPERBLOCK>>>(dev_a, dev_b, dev_c);

	// Get stop time event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// Compute execution time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU time: %13f msec\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost );

	// verify that the GPU did the work we requested 
	bool success = true;
	for (int i=0; i<N; i++){
		if((a[i] + b[i] != c[i])){
			printf( "Error: %d + %d != %d \n", a[i], b[i], c[i] );
			success = false;
		}
	}
	if (success) printf("We did it !~\n");


	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
