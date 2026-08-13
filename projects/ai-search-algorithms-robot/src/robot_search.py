import numpy as np
import matplotlib.pyplot as plt

N = 10
matriz = np.zeros((N, N))

obstáculos = [(1, 2), (1, 3), (1, 4), (2, 7), (3, 4), (5, 4), (2, 8), (3, 8), (4, 2), (5, 3), (7, 3), (7, 4),
              (6, 0), (7, 0), (8, 0), (9, 0), (7, 5), (7, 8), (8, 8), (9, 4)]
for obs in obstáculos:
    matriz[obs] = 1

posición_inicial = (0, 0)
posición_final = (9, 9)
matriz[posición_inicial] = 2
matriz[posición_final] = 3

def obtener_vecinos(matriz, cell):
    vecinos = []
    direcciones = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in direcciones:
        nuevo_x, nuevo_y = cell[0] + dx, cell[1] + dy
        if 0 <= nuevo_x < N and 0 <= nuevo_y < N and matriz[nuevo_x][nuevo_y] != 1:
            vecinos.append((nuevo_x, nuevo_y))
    return vecinos

def profundidad(inicio, end):
    pila = [(inicio, [inicio])]
    while pila:
        actual, path = pila.pop()
        if actual == end:
            return path
        for vecino in obtener_vecinos(matriz, actual):
            if vecino not in path:
                pila.append((vecino, path + [vecino]))
    return []

def busqueda_haz(inicio, end, beam_width=2):
    heap = [(0, [inicio])]
    while heap:
        heap.sort(key=lambda x: x[0])
        heap = heap[:beam_width]
        _, path = heap.pop(0)
        actual = path[-1]
        if actual == end:
            return path
        for vecino in obtener_vecinos(matriz, actual):
            if vecino not in path:
                new_path = path + [vecino]
                cost = len(new_path) + np.linalg.norm(np.array(vecino) - np.array(end))
                heap.append((cost, new_path))
    return []

def busqueda_a(inicio, end):
    lista = [(0, inicio)]
    came_from = {}
    g_score = {cell: float('inf') for row in matriz for cell in row}
    g_score[inicio] = 0
    
    while lista:
        lista.sort(key=lambda x: x[0])
        _, actual = lista.pop(0)
        
        if actual == end:
            path = []
            while actual in came_from:
                path.insert(0, actual)
                actual = came_from[actual]
            return path
        
        for vecino in obtener_vecinos(matriz, actual):
            tentative_g_score = g_score[actual] + 1
            if vecino not in g_score or tentative_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g_score
                f_score = tentative_g_score + np.linalg.norm(np.array(vecino) - np.array(end))
                lista.append((f_score, vecino))
    
    return []

def main():
    print("Seleccione el tipo de búsqueda:")
    print("1. Profundidad")
    print("2. Busqueda haz")
    print("3. A*")
    
    opcion = input("Ingrese el número de la búsqueda deseada: ")
    
    if opcion == '1':
        ruta = profundidad(posición_inicial, posición_final)
        print("Ruta Profundidad:", ruta)
    elif opcion == '2':
        ruta = busqueda_haz(posición_inicial, posición_final)
        print("Ruta busqueda haz:", ruta)
    elif opcion == '3':
        ruta = busqueda_a(posición_inicial, posición_final)
        print("Ruta A*:", ruta)
    else:
        print("Opción no válida.")

    cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'blue', 'red'])
    plt.figure(figsize=(8, 8))
    plt.imshow(matriz, cmap=cmap, origin='upper')

    if ruta:
        ruta_x, ruta_y = zip(*ruta)
        plt.plot(ruta_y, ruta_x, marker='o', markersize=8, color='green', label='Ruta')
        plt.legend()

    plt.title('Tablero ROBOT')
    plt.xlabel('Columnas')
    plt.ylabel('Filas')
    plt.show()

if __name__ == "__main__":
    main()
