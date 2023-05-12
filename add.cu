#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c){
	*c = *a + *b;
}


int main(void){
	int ha=1, hb=2, hc;
	//add<<<1,1>>>(&ha, &hb, &hc);
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, sizeof(int));
	cudaMalloc((void **)&d_b, sizeof(int));
	cudaMalloc((void **)&d_c, sizeof(int));

	cudaMemcpy(d_a, &ha, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &hb, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &hc, sizeof(int), cudaMemcpyHostToDevice);

	add<<<1,1>>>(d_a, d_b, d_c);

	cudaMemcpy(&hc, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("hc = %d\n", hc);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("hc = %d\n", hc);


	return 0;
}
