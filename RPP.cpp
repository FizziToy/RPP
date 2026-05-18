#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 3000;

    vector<double> A;
    vector<double> b(N);
    vector<double> x(N);

    vector<int> rows(size);
    vector<int> displsRows(size);
    vector<int> sendCountsA(size);
    vector<int> displsA(size);

    int baseRows = N / size;
    int remainder = N % size;

    for (int p = 0; p < size; p++)
    {
        rows[p] = baseRows + (p < remainder ? 1 : 0);
    }

    displsRows[0] = 0;
    for (int p = 1; p < size; p++)
    {
        displsRows[p] = displsRows[p - 1] + rows[p - 1];
    }

    for (int p = 0; p < size; p++)
    {
        sendCountsA[p] = rows[p] * N;
        displsA[p] = displsRows[p] * N;
    }

    int localRows = rows[rank];
    int firstRow = displsRows[rank];

    vector<double> localA(localRows * N);
    vector<double> localB(localRows);

    if (rank == 0)
    {
        A.resize(N * N);

        srand(time(0));

        for (int i = 0; i < N; i++)
        {
            double sum = 0;

            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = rand() % 10 + 1;
                sum += abs(A[i * N + j]);
            }

            A[i * N + i] = sum + 1;
            b[i] = rand() % 100;
        }
    }

    double start = MPI_Wtime();

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        sendCountsA.data(),
        displsA.data(),
        MPI_DOUBLE,
        localA.data(),
        localRows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    vector<int> sendCountsB(size);
    vector<int> displsB(size);

    for (int p = 0; p < size; p++)
    {
        sendCountsB[p] = rows[p];
        displsB[p] = displsRows[p];
    }

    MPI_Scatterv(
        rank == 0 ? b.data() : nullptr,
        sendCountsB.data(),
        displsB.data(),
        MPI_DOUBLE,
        localB.data(),
        localRows,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    vector<double> pivotRow(N);
    double pivotB;

    for (int k = 0; k < N; k++)
    {
        int owner = 0;

        for (int p = 0; p < size; p++)
        {
            if (k >= displsRows[p] && k < displsRows[p] + rows[p])
            {
                owner = p;
                break;
            }
        }

        if (rank == owner)
        {
            int localIndex = k - firstRow;

            for (int j = 0; j < N; j++)
            {
                pivotRow[j] = localA[localIndex * N + j];
            }

            pivotB = localB[localIndex];
        }

        MPI_Bcast(pivotRow.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivotB, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (int i = 0; i < localRows; i++)
        {
            int globalRow = firstRow + i;

            if (globalRow > k)
            {
                double factor = localA[i * N + k] / pivotRow[k];

                for (int j = k; j < N; j++)
                {
                    localA[i * N + j] -= factor * pivotRow[j];
                }

                localB[i] -= factor * pivotB;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Gatherv(
        localA.data(),
        localRows * N,
        MPI_DOUBLE,
        rank == 0 ? A.data() : nullptr,
        sendCountsA.data(),
        displsA.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gatherv(
        localB.data(),
        localRows,
        MPI_DOUBLE,
        rank == 0 ? b.data() : nullptr,
        sendCountsB.data(),
        displsB.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        for (int i = N - 1; i >= 0; i--)
        {
            x[i] = b[i];

            for (int j = i + 1; j < N; j++)
            {
                x[i] -= A[i * N + j] * x[j];
            }

            x[i] /= A[i * N + i];
        }
    }

    double end = MPI_Wtime();

    if (rank == 0)
    {
        cout << "Matrix size: " << N << endl;
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << end - start << " sec" << endl;
    }

    MPI_Finalize();

    return 0;
}