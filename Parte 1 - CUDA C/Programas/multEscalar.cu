#include "stdio.h"
#define N 100

__global__ void add(int *a, int *c)
{
  int tID   = blockIdx.x;
  if (tID < N)
  {
	c[tID] = 3*a[tID];
  }
}

int main()
{
	
  int a[N], c[N];
  int *d_a, *d_c;

  cudaMalloc((void **) &d_a, N*sizeof(int));
  cudaMalloc((void **) &d_c, N*sizeof(int));

  // Llenar el arreglo
  for (int i = 0; i < N; i++)
  {
	a[i] = i;
  }

  cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

  add<<<N,1>>>(d_a, d_c);

  cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
	printf("3*%d = %d\n", a[i], c[i]);
  }
  
  return 0;
  
}
