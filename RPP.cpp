#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 2000;

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

    ofstream logFile;

    if (rank == 0)
    {
        logFile.open("mpi_log.txt");

        logFile << "MPI Gauss method log" << endl;
        logFile << "Matrix size: " << N << endl;
        logFile << "Processes: " << size << endl << endl;

        logFile << "Rows distribution:" << endl;

        for (int p = 0; p < size; p++)
        {
            logFile << "Process " << p
                << " takes rows from " << displsRows[p]
                << " to " << displsRows[p] + rows[p] - 1
                << " (" << rows[p] << " rows)" << endl;
        }

        logFile << endl;
    }

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
    double pivotB = 0;

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

        if (rank == 0 && k < 10)
        {
            logFile << "----------------------------------------" << endl;
            logFile << "Step k = " << k << endl;
            logFile << "Pivot row owner: process " << owner << endl;
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

        if (rank == 0 && k < 10)
        {
            logFile << "Pivot row was broadcasted from process " << owner << endl;
            logFile << "Pivot element A[" << k << "][" << k << "] = " << pivotRow[k] << endl;
            logFile << "Pivot b[" << k << "] = " << pivotB << endl << endl;
        }

        stringstream localLog;

        for (int i = 0; i < localRows; i++)
        {
            int globalRow = firstRow + i;

            if (globalRow > k)
            {
                double oldB = localB[i];
                double oldAik = localA[i * N + k];

                double factor = localA[i * N + k] / pivotRow[k];

                for (int j = k; j < N; j++)
                {
                    localA[i * N + j] -= factor * pivotRow[j];
                }

                localB[i] -= factor * pivotB;

                if (k < 10 && globalRow < k + 10)
                {
                    localLog << "Process " << rank
                        << " processed row " << globalRow
                        << ": factor = " << factor
                        << ", A[" << globalRow << "][" << k << "] "
                        << oldAik << " -> " << localA[i * N + k]
                        << ", b[" << globalRow << "] "
                        << oldB << " -> " << localB[i]
                        << endl;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        string localLogStr = localLog.str();
        int localLogSize = localLogStr.size();

        vector<int> logSizes(size);
        vector<int> logDispls(size);

        MPI_Gather(
            &localLogSize,
            1,
            MPI_INT,
            logSizes.data(),
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        vector<char> allLogs;

        if (rank == 0)
        {
            logDispls[0] = 0;

            for (int p = 1; p < size; p++)
            {
                logDispls[p] = logDispls[p - 1] + logSizes[p - 1];
            }

            int totalLogSize = 0;

            for (int p = 0; p < size; p++)
            {
                totalLogSize += logSizes[p];
            }

            allLogs.resize(totalLogSize);
        }

        MPI_Gatherv(
            localLogStr.data(),
            localLogSize,
            MPI_CHAR,
            rank == 0 ? allLogs.data() : nullptr,
            logSizes.data(),
            logDispls.data(),
            MPI_CHAR,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0 && k < 10)
        {
            string resultLog(allLogs.begin(), allLogs.end());
            logFile << resultLog << endl;
        }
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
        ofstream resultFile("result.txt");

        resultFile << "Matrix size: " << N << endl;
        resultFile << "Processes: " << size << endl;
        resultFile << "Execution time: " << end - start << " sec" << endl << endl;

        resultFile << "Result vector x:" << endl;

        for (int i = 0; i < N; i++)
        {
            resultFile << "x[" << i << "] = " << x[i] << endl;
        }

        resultFile.close();

        logFile.close();

        cout << "Matrix size: " << N << endl;
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << end - start << " sec" << endl;
        cout << "Results were written to result.txt" << endl;
        cout << "Log was written to mpi_log.txt" << endl;
    }

    MPI_Finalize();

    return 0;
}