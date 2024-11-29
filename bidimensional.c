#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SRAND_VALUE 1985

// Função para inicializar o tabuleiro com valores aleatórios
void inicializa_tabuleiro(int **grid, int N, int rank, int dims[2], int coords[2]) {
    srand(SRAND_VALUE + rank);  // Diferente seed para cada processo
    int rows_per_process = N / dims[0];
    int cols_per_process = N / dims[1];
    
    int start_row = coords[0] * rows_per_process;
    int end_row = (coords[0] + 1) * rows_per_process;
    int start_col = coords[1] * cols_per_process;
    int end_col = (coords[1] + 1) * cols_per_process;

    for (int i = start_row; i < end_row; i++) {
        for (int j = start_col; j < end_col; j++) {
            grid[i][j] = rand() % 2;
        }
    }
}

// Função para contar os vizinhos vivos de uma célula
int getNeighbors(int **grid, int N, int i, int j) {
    int neighbors = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = (i + di + N) % N;
            int nj = (j + dj + N) % N;
            neighbors += grid[ni][nj];
        }
    }
    return neighbors;
}

// Função que computa a próxima geração para um processo
void computa_geracao(int **grid, int **newgrid, int N, int rank, int dims[2], int coords[2]) {
    int rows_per_process = N / dims[0];
    int cols_per_process = N / dims[1];
    
    int start_row = coords[0] * rows_per_process;
    int end_row = (coords[0] + 1) * rows_per_process;
    int start_col = coords[1] * cols_per_process;
    int end_col = (coords[1] + 1) * cols_per_process;

    for (int i = start_row; i < end_row; i++) {
        for (int j = start_col; j < end_col; j++) {
            int neighbors = getNeighbors(grid, N, i, j);
            if (grid[i][j] == 1) {
                if (neighbors < 2 || neighbors > 3) {
                    newgrid[i][j] = 0; // Morre por abandono ou superpopulação
                } else {
                    newgrid[i][j] = 1; // Sobrevive
                }
            } else {
                if (neighbors == 3) {
                    newgrid[i][j] = 1; // Nascimento
                } else {
                    newgrid[i][j] = 0; // Continua morta
                }
            }
        }
    }
}

// Função que troca as bordas entre processos (usando MPI_Sendrecv)
void troca_bordas(int **grid, int **newgrid, int N, MPI_Comm comm_cart, int dims[2], int coords[2]) {
    int up, down, left, right;
    MPI_Status status[4]; // Status das requisições

    // Determina os vizinhos usando a topologia cartesiana
    MPI_Cart_shift(comm_cart, 0, 1, &up, &down);    // Troca na direção das linhas (norte/sul)
    MPI_Cart_shift(comm_cart, 1, 1, &left, &right);  // Troca na direção das colunas (leste/oeste)

    // Envia e recebe as bordas para as direções norte e sul
    MPI_Sendrecv(grid[up], N / dims[1], MPI_INT, up, 0, newgrid[down], N / dims[1], MPI_INT, down, 0, comm_cart, &status[0]);
    MPI_Sendrecv(grid[down], N / dims[1], MPI_INT, down, 0, newgrid[up], N / dims[1], MPI_INT, up, 0, comm_cart, &status[1]);

    // Envia e recebe as bordas para as direções leste e oeste
    MPI_Sendrecv(grid[left], N / dims[0], MPI_INT, left, 0, newgrid[right], N / dims[0], MPI_INT, right, 0, comm_cart, &status[2]);
    MPI_Sendrecv(grid[right], N / dims[0], MPI_INT, right, 0, newgrid[left], N / dims[0], MPI_INT, left, 0, comm_cart, &status[3]);
}

// Função para contar as células vivas
int contar_celulas_vivas(int **grid, int N) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            count += grid[i][j];
        }
    }
    return count;
}

int main(int argc, char **argv) {
    int N = 2048;  // Dimensão do tabuleiro
    int geracoes = 2000; // Número de gerações
    int rank, num_processos;
    int dims[2] = {0, 0};  // Dimensões da topologia cartesiana
    int coords[2];         // Coordenadas do processo na topologia cartesiana

    // Inicializa o MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    // Cria a topologia cartesiana 2D
    MPI_Dims_create(num_processos, 2, dims); // Cria 2D layout (dimensões do grid)
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, (int[2]){1, 1}, 0, &comm_cart); // Cria comunicador cartesiano 2D
    MPI_Cart_coords(comm_cart, rank, 2, coords);  // Obtém as coordenadas do processo

    // Aloca as matrizes para o tabuleiro
    int **grid = (int **)malloc(N * sizeof(int *));
    int **newgrid = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        grid[i] = (int *)malloc(N * sizeof(int));
        newgrid[i] = (int *)malloc(N * sizeof(int));
    }

    // Inicializa o tabuleiro em paralelo
    inicializa_tabuleiro(grid, N, rank, dims, coords);

    // Contando as células vivas no processo local
    int celulas_vivas_local = contar_celulas_vivas(grid, N);

    // Usando MPI_Reduce para somar as células vivas de todos os processos
    int celulas_vivas_total;
    MPI_Reduce(&celulas_vivas_local, &celulas_vivas_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // O processo 0 imprime o total de células vivas
    if (rank == 0) {
        printf("Total de células vivas no início: %d\n", celulas_vivas_total);
    }

    // Medindo o tempo de execução
    double start_time = MPI_Wtime();

    // Executando as gerações
    for (int g = 0; g < geracoes; g++) {
        troca_bordas(grid, newgrid, N, comm_cart, dims, coords); // Troca as bordas
        computa_geracao(grid, newgrid, N, rank, dims, coords); // Computa a nova geração

        // Troca os grids
        int **temp = grid;
        grid = newgrid;
        newgrid = temp;
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // Contando as células vivas após as gerações
    int celulas_vivas = contar_celulas_vivas(grid, N);
    printf("Processo %d: Células vivas após %d gerações: %d\n", rank, geracoes, celulas_vivas);
    printf("Tempo total de execução no processo %d: %f segundos\n", rank, total_time);

    // Finaliza o MPI
    MPI_Finalize();

    // Liberando memória
    for (int i = 0; i < N; i++) {
        free(grid[i]);
        free(newgrid[i]);
    }
    free(grid);
    free(newgrid);

    return 0;
}
