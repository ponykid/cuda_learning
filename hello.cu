#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello( void ){
	printf("hello, cuda\n");
}

__global__ void helloCuda( float *a ){
	printf("hello ! data = %f \n", *a);
}

int main( void ){
	printf("hello world from CPU \n");

	// kernel_function<<<num_blocks, num_thread>>>(param1,param2)
	hello<<<1,10>>>();

	cudaDeviceReset();

	float h_a = 1;
	float *d_a;
	cudaMalloc(&d_a, sizeof(float));
	cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice);

	helloCuda<<<1,10>>>(d_a);
	cudaMemcpy(&h_a, d_a, sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceReset();
	//cudaDeviceSynchronize();

	cudaFree(&d_a);

	int size = 1<<24;

	printf("size = %d\n", size);

	return 0;
}
