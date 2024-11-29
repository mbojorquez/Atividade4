#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define SRAND_VALUE 1985

// Função para inicializar o tabuleiro com valores aleatórios
void inicializa_tabuleiro(int **grid, int N, int rank, int num_processos) {
    srand(SRAND_VALUE + rank);  // Diferente seed para cada processo
    int start_row = rank * (N / num_processos);
    int end_row = (rank + 1) * (N / num_processos);
    if (rank == num_processos - 1) {
        end_row = N;  // Último processo pega as linhas restantes
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
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
void computa_geracao(int **grid, int **newgrid, int N, int rank, int num_processos) {
    int start_row = rank * (N / num_processos);
    int end_row = (rank + 1) * (N / num_processos);
    if (rank == num_processos - 1) {
        end_row = N; // Último processo
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
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
void troca_bordas(int **grid, int **newgrid, int N, int rank, int num_processos) {
    int above, below;
    MPI_Status status[4]; // Status das requisições

    // Determina o rank do processo acima e abaixo
    above = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    below = (rank == num_processos - 1) ? MPI_PROC_NULL : rank + 1;

    // Envia a linha superior e recebe a linha inferior
    if (above != MPI_PROC_NULL) {
        MPI_Sendrecv(grid[rank * (N / num_processos)], N, MPI_INT, above, 0, 
                     newgrid[(rank * (N / num_processos)) - 1], N, MPI_INT, above, 0, 
                     MPI_COMM_WORLD, &status[0]);
    }

    // Envia a linha inferior e recebe a linha superior
    if (below != MPI_PROC_NULL) {
        MPI_Sendrecv(grid[(rank + 1) * (N / num_processos) - 1], N, MPI_INT, below, 0, 
                     newgrid[((rank + 1) * (N / num_processos))], N, MPI_INT, below, 0, 
                     MPI_COMM_WORLD, &status[1]);
    }
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
    int geracoes = 2000; // Corrigido: alterado "gerações" para "geracoes"
    int rank, num_processos;

    // Inicializa o MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    // Aloca as matrizes para o tabuleiro
    int **grid = (int **)malloc(N * sizeof(int *));
    int **newgrid = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        grid[i] = (int *)malloc(N * sizeof(int));
        newgrid[i] = (int *)malloc(N * sizeof(int));
    }

    // Inicializa o tabuleiro em paralelo
    inicializa_tabuleiro(grid, N, rank, num_processos);

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
        troca_bordas(grid, newgrid, N, rank, num_processos); // Troca as bordas
        computa_geracao(grid, newgrid, N, rank, num_processos); // Computa a nova geração

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
