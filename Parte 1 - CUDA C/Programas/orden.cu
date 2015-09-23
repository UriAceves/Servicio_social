#include <stdio.h>

#define NUM_BLOCKS 16
#define ANCHURA_BLOCK 1

__global__ void hola()
{
    printf("Hola mundo! Soy un thread en el bloque %d\n", blockIdx.x);
}


int main(int argc,char **argv)
{
    // lanzar el kernel
    hola<<<NUM_BLOCKS, ANCHURA_BLOCK>>>();

    // forzar a los printf() para que se muestren
    cudaDeviceSynchronize();

    printf("Eso es todo amigos!\n");

    return 0;
}