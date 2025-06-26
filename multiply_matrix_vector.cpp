#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>


void initializeMatrix(std::vector<double>& matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = static_cast<double>(i + j + 1);
        }
    }
}


void initializeVector(std::vector<double>& vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = static_cast<double>(i + 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m = 4;
    int n = 6;
    int k_cols_per_process;

    if (n % world_size != 0) {
        if (world_rank == 0) {
            std::cerr << "Помилка: Кількість стовпців матриці (n) повинна бути кратною кількості процесів." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    k_cols_per_process = n / world_size;

    std::vector<double> A; // Матриця
    std::vector<double> x; // Вхідний вектор
    std::vector<double> y(m, 0.0); // Результат множення A * x

    if (world_rank == 0) {
        A.resize(m * n);
        initializeMatrix(A, m, n);
        x.resize(n);
        initializeVector(x, n);

        std::cout << "Матриця A (" << m << "x" << n << "):" << std::endl;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << A[i * n + j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Вектор x (" << n << "):" << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << x[i] << "\t";
        }
        std::cout << std::endl << std::endl;
    }

    x.resize(n); // Всі процеси мають отримати вектор x
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    std::vector<double> local_partial_y(m, 0.0); // Частковий результат для кожного процесу

    int start_col = world_rank * k_cols_per_process;
    int end_col = start_col + k_cols_per_process;


    for (int i = 0; i < m; ++i) { // Рядки матриці
        for (int j = start_col; j < end_col; ++j) { // Стовпці, що належать цьому процесу

            double val_A_ij = static_cast<double>(i + j + 1);
            local_partial_y[i] += val_A_ij * x[j];
        }
    }

    MPI_Allreduce(local_partial_y.data(), y.data(), m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Результат множення A * x:" << std::endl;
        for (int i = 0; i < m; ++i) {
            std::cout << y[i] << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
