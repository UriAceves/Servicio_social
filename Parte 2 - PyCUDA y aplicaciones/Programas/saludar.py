import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

    __global__ void saluda()
    {
      printf("Mi indice x es %d, mi indice en y es %d\\n", threadIdx.x, threadIdx.y);
    }
    """)

func = mod.get_function("saluda")
func(block=(4,4,1))