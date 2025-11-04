#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cmath>

double definite_solution(double x, double y) {
    return sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
}

double definite_derivative(double x, double y) {
    return -2.0 * (2.0 * M_PI) * (2.0 * M_PI) * sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
}

void run_solver(int nx, int ny, int rank, int size, FILE *data_file) {
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double dx = (x_max - x_min) / (nx - 1);
    double dy = (y_max - y_min) / (ny - 1);
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coeff = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2));

    double tolerance = 1.0e-6;
    int max_iterations = 100000;

    int local_nx = nx / size;
    int remainder = nx % size;
    int start_row = rank * local_nx + (rank < remainder ? rank : remainder);
    local_nx += (rank < remainder ? 1 : 0);
    int local_nx_with_ghosts = local_nx + 2;

    double **u = (double **)malloc(local_nx_with_ghosts * sizeof(double *));
    double **u_new = (double **)malloc(local_nx_with_ghosts * sizeof(double *));
    double **f = (double **)malloc(local_nx_with_ghosts * sizeof(double *));
    double **u_exact = (double **)malloc(local_nx_with_ghosts * sizeof(double *));
    for (int i = 0; i < local_nx_with_ghosts; i++) {
        u[i] = (double *)calloc(ny, sizeof(double));
        u_new[i] = (double *)calloc(ny, sizeof(double));
        f[i] = (double *)calloc(ny, sizeof(double));
        u_exact[i] = (double *)calloc(ny, sizeof(double));
    }

    // Initialize arrays
    for (int i = 1; i <= local_nx; i++) {
        int global_i = start_row + i - 1;
        for (int j = 0; j < ny; j++) {
            double x = x_min + global_i * dx;
            double y = y_min + j * dy;
            f[i][j] = definite_derivative(x, y);
            u_exact[i][j] = definite_solution(x, y);
        }
    }

    // Boundary conditions
    if (start_row == 0) {
        for (int j = 0; j < ny; j++) {
            u[1][j] = u_new[1][j] = 0.0;
        }
    }
    if (start_row + local_nx == nx) {
        for (int j = 0; j < ny; j++) {
            u[local_nx][j] = u_new[local_nx][j] = 0.0;
        }
    }

    int iter;
    double error = 0.0;
    double start_time = MPI_Wtime();

    MPI_Request req[4];
    int req_count = 0;

    for (iter = 0; iter < max_iterations; iter++) {
        // Exchange ghost rows with neighbors using non-blocking sends
        req_count = 0;

        if (rank > 0) {
            MPI_Isend(u[1], ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[req_count++]);
            MPI_Irecv(u[0], ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[req_count++]);
        }
        if (rank < size-1) {
            MPI_Isend(u[local_nx], ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[req_count++]);
            MPI_Irecv(u[local_nx+1], ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[req_count++]);
        }

        // Wait for all communication to complete
        if (req_count > 0) {
            MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
        }

        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j < ny-1; j++) {
                u_new[i][j] = coeff * (
                    (u[i-1][j] + u[i+1][j]) / dx2 +
                    (u[i][j-1] + u[i][j+1]) / dy2 -
                    f[i][j]
                );
            }
        }

        double local_error = 0.0;
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j < ny-1; j++) {
                double diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > local_error) local_error = diff;
            }
        }

        // Global max error
        MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Swap pointers
        double **tmp = u; u = u_new; u_new = tmp;

        if (error < tolerance) {
            iter++;
            break;
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        int interior_points = (nx - 2) * (ny - 2);
        long long flops_per_iter = interior_points * 10;
        long long total_flops = flops_per_iter * iter;
        double gflops = (total_flops / elapsed_time) / 1.0e9;

        long long bytes_per_iter = interior_points * 10 * sizeof(double);
        long long total_bytes = bytes_per_iter * iter;
        double mem_bandwidth_gb = (total_bytes / elapsed_time) / 1.0e9;
        double arithmetic_intensity = (double)total_flops / (double)total_bytes;

        fprintf(data_file, "%d,%d,%d,%lld,%.6f,%.6f,%.6f,%.6f\n",
                nx, ny, iter, total_flops, elapsed_time, gflops,
                arithmetic_intensity, mem_bandwidth_gb);

        printf("Converged in %d iterations, time = %.6f s, GFLOPS = %.4f\n\n",
               iter, elapsed_time, gflops);
    }

    for (int i = 0; i < local_nx_with_ghosts; i++) {
        free(u[i]);
        free(u_new[i]);
        free(f[i]);
        free(u_exact[i]);
    }
    free(u);
    free(u_new);
    free(f);
    free(u_exact);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *data_file = NULL;
    if (rank == 0) {
        data_file = fopen("performance_data.csv", "w");
        if (!data_file) {
            fprintf(stderr, "Error opening performance_data.csv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(data_file, "nx,ny,iters,FLOPS,Time,GFLOPS,AI,MemBW\n");
    }

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int nx = sizes[s];
        int ny = sizes[s];
        run_solver(nx, ny, rank, size, data_file);
    }

    if (rank == 0) fclose(data_file);

    MPI_Finalize();
    return 0;
}
