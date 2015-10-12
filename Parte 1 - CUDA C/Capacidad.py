############################ Modulo para calcular la capacidad de un kernel con respecto   ##################################
############################ a la dimension de los bloques                                 ##################################
############################ Se utiliza unicamente en el notebook 5                        ##################################

from matplotlib import pyplot as plt
import numpy as np

def capacidad(num_max_threads_total, num_max_threads_bloque):

    N = int(np.sqrt(num_max_threads_bloque))
    Capacidad = np.zeros(N)
    Bloques = np.zeros(N)

    for i in xrange(N):

        n = i+1
        num_threads_bloque = n*n
        threads_usados = 8*num_threads_bloque

        bloques_utilizados = num_max_threads_total/num_threads_bloque

        if bloques_utilizados > 8:
            Capacidad[i] = (8.*num_threads_bloque)/num_max_threads_total
            Bloques[i] = 8

        elif bloques_utilizados == 8:
            Capacidad[i] = 1.
            Bloques[i] = 8

        elif bloques_utilizados < 8:
            Capacidad[i] = (float(bloques_utilizados)*num_threads_bloque)/num_max_threads_total
            Bloques[i] = bloques_utilizados

    plt.figure(figsize=(10,10))
    plt.plot([i+1 for i in xrange(N)], Capacidad)
    plt.title('Capacidad total con respecto a la dimension n de los bloques (nxn)')
    plt.xlim(1,N+1)
    plt.xlabel("n")
    plt.ylim(0.,1.1)
    plt.ylabel("Capacidad")
    plt.grid()

    plt.figure(figsize=(10,10))
    plt.plot([i+1 for i in xrange(N)], Bloques)
    plt.title("Bloques utilizados con respecto a la dimension n de los bloques (nxn)")
    plt.xlim(1,N+1)
    plt.xlabel("n")
    plt.ylim(1, 10)
    plt.ylabel("Bloques")
    plt.grid()