#include "libs/cudarender.h"

__global__ void printHelloWorld()
{
	printf("Hello, World!\n");
}

int main(int argc, char const *argv[])
{
	printHelloWorld<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
