#include <stdio.h>
#define DIM_MAX 768

// Creamos el kernel. Es interesante notar como un arreglo 2D en el host se vuelve 1D en el device. 
__global__ void suma(int * d_arreglo){

	int idx_x = blockIdx.x*blockDim.x + threadIdx.x ;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y ;

	for (int t = 0; t < 1000; t ++) {
		if (idx_x*blockDim.x + idx_y < 1536) {
		  d_arreglo[idx_x*blockDim.x + idx_y] += 1 ;
		  __syncthreads() ;
		}
	}
	
}

int main(int argc, char ** argv){

	// Definimos los arreglos de entrada, salida y aquel que ira en el device
	int arreglo[DIM_MAX][DIM_MAX] ;
	int arreglo_salida[DIM_MAX][DIM_MAX] ;
	int * d_arreglo ;

	// Colocamos las condiciones iniciales. Un arreglo lleno de 0's
	for (int x = 0; x < DIM_MAX; x ++) {
		for (int y = 0; y < DIM_MAX; y ++) {
			arreglo[x][y] = 0 ;
		}
	}

	// Alojamos d_arreglo en el device
	cudaMalloc((void**) &d_arreglo, DIM_MAX*DIM_MAX*sizeof(int)) ;				
	
	// Aqui introducimos un tipo de variable nuevo. Los detalles no son importantes, pero seran los "Events"
	// que nos permitiran medir el tiempo
	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;

	// Se copia la memoria en el device desde el host
	cudaMemcpy(d_arreglo, arreglo, DIM_MAX*DIM_MAX*sizeof(int), cudaMemcpyHostToDevice) ;
	
	// Definimos la dimension de la malla y los bloques
	int dimB = atoi(argv[1]) ; int dimG = atoi(argv[2]) ; 
	dim3 dimBlock(dimB, dimB, 1) ;
	dim3 dimGrid(dimG, 1, 1) ;

	// Empezamos a medir el tiempo
	cudaEventRecord(start) ;

	// Lanzamos el kernel
	suma<<<dimGrid, dimBlock>>>(d_arreglo) ;
	
	// Terminamos de medir el tiempo
	cudaEventRecord(stop) ;
	
	// Copiamos de regreso
	cudaMemcpy(arreglo_salida, d_arreglo, DIM_MAX*DIM_MAX*sizeof(int), cudaMemcpyDeviceToHost) ;

	// Y terminamos con los eventos. La sincronizacion permite que solo el device se desempeñe sin tener
	// que preocuparse por el host
	cudaEventSynchronize(stop);
	
	// calculamos el tiempo. Esto gracias a cudaEventElapsedTime, la cual es una API ya integrada en CUDA
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	// Calculamos el resultado final
	int resultado = 0 ;
	for (int x = 0; x < DIM_MAX; x ++) {
		for (int y = 0; y < DIM_MAX; y ++) {
			 resultado += arreglo_salida[x][y] ;
		}
	}
	
	// Imprimimos los resultados
	printf("El resultado final es %d\n", resultado) ;
 	printf("Tiempo de ejecucion del kernel %f ms\n", milliseconds) ;
	
	cudaFree(d_arreglo) ;

	return 0 ;
}
