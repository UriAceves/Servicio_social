#include <stdio.h>

#define DIM_MAX 768

__global__ void suma(int * d_arreglo){

	int idx_x = blockIdx.x*blockDim.x + threadIdx.x ;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y ;

	for (int t = 0; t < 1000; t ++) {
		d_arreglo[idx_x*blockDim.x + idx_y] += 1 ;
		__syncthreads() ;
	}
	
}

int main(int argc, char ** argv){

	int arreglo[DIM_MAX][DIM_MAX] ;
	int arreglo_salida[DIM_MAX][DIM_MAX] ;
	int * d_arreglo ;


	for (int x = 0; x < DIM_MAX; x ++) {
		for (int y = 0; y < DIM_MAX; y ++) {
			arreglo[x][y] = 0 ;
		}
	}


	cudaMalloc((void**) &d_arreglo, DIM_MAX*DIM_MAX*sizeof(int)) ;				
	
	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;

	cudaMemcpy(d_arreglo, arreglo, DIM_MAX*DIM_MAX*sizeof(int), 
cudaMemcpyHostToDevice) ;
	
	int dimB = atoi(argv[1]) ; int dimG = atoi(argv[2]) ; 
	dim3 dimBlock(dimB, dimB, 1) ;
	dim3 dimGrid(dimG, 1, 1) ;

	cudaEventRecord(start) ;

	suma<<<dimGrid, dimBlock>>>(d_arreglo) ;

	cudaEventRecord(stop) ;

	cudaMemcpy(arreglo_salida, d_arreglo, DIM_MAX*DIM_MAX*sizeof(int), 
cudaMemcpyDeviceToHost) ;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	int resultado = 0 ;
	for (int x = 0; x < DIM_MAX; x ++) {
		for (int y = 0; y < DIM_MAX; y ++) {
			 resultado += arreglo_salida[x][y] ;
		}
	}
	
	printf("El resultado final es %d\n", resultado) ;
 	printf("Tiempo de ejecucion del kernel %f ms\n", milliseconds) ;
	
	//printf(“Tiempo de ejecucion del kernel: %f ms\n”, milliseconds) ;
	cudaFree(d_arreglo) ;

	return 0 ;
}
