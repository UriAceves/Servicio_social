#include <stdio.h>

__global__ void elevar_al_cuadrado(float * d_salida, float * d_entrada){
	int idx = threadIdx.x;
    float f = d_entrada[idx];
    d_salida[idx] = f*f;
}

int main(int argc, char ** argv) {

	const int TAMANIO_ARREGLO = 100;
	const int BYTES_ARREGLO = TAMANIO_ARREGLO * sizeof(float);

	// Generamos el arreglo de entrada en el anfitrion
	float h_entrada[TAMANIO_ARREGLO];
    
	for (int i = 0; i < TAMANIO_ARREGLO; i++) {
		h_entrada[i] = float(i);
	}
    
	float h_salida[TAMANIO_ARREGLO];

	// Declaramos apuntadores de memoria en GPU
	float * d_entrada;
	float * d_salida;

	// Reservamos memoria del GPU
	cudaMalloc((void**) &d_entrada, BYTES_ARREGLO);
	cudaMalloc((void**) &d_salida, BYTES_ARREGLO);

	// Copiamos informacion al GPU
	cudaMemcpy(d_entrada, h_entrada, BYTES_ARREGLO, cudaMemcpyHostToDevice);

	// Lanza el kernel
	elevar_al_cuadrado<<<1, TAMANIO_ARREGLO>>>(d_salida, d_entrada);

	// Copiamos el arreglo resultante al GPU
	cudaMemcpy(h_salida, d_salida, BYTES_ARREGLO, cudaMemcpyDeviceToHost);

	// Imprimimos el arreglo resultante
	for (int i =0; i < TAMANIO_ARREGLO; i++) {
		printf("%f", h_salida[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_entrada);
	cudaFree(d_salida);

	return 0;
}
