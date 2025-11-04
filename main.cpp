#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cmath>

double definite_solution(double x, double y) {
    return sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
}

double definite_derivative(double x, double y) {
    return -2.0 * (2.0 * M_PI) * (2.0 * M_PI) * sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
}

void run_solver(int nx, int ny, FILE *data_file) {
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    double dx = (x_max - x_min) / (nx - 1);
    double dy = (y_max - y_min) / (ny - 1);
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coeff = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2));

    double tolerance = 1.0e-6;
    int max_iterations = 100000;

    double **u = (double **)malloc(nx * sizeof(double *));
    double **u_new = (double **)malloc(nx * sizeof(double *));
    double **f = (double **)malloc(nx * sizeof(double *));
    double **u_exact = (double **)malloc(nx * sizeof(double *));
    for (int i = 0; i < nx; i++) {
        u[i] = (double *)malloc(ny * sizeof(double));
        u_new[i] = (double *)malloc(ny * sizeof(double));
        f[i] = (double *)malloc(ny * sizeof(double));
        u_exact[i] = (double *)malloc(ny * sizeof(double));
    }

    // Initialize arrays
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = x_min + i * dx;
            double y = y_min + j * dy;
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
            f[i][j] = definite_derivative(x, y);
            u_exact[i][j] = definite_solution(x, y);
        }
    }

    // Boundary conditions
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        u[i][0] = u[i][ny-1] = u_new[i][0] = u_new[i][ny-1] = 0.0;
    }
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        u[0][j] = u[nx-1][j] = u_new[0][j] = u_new[nx-1][j] = 0.0;
    }

    printf("=== Grid %d x %d ===\n", nx, ny);

    int iter;
    double error = 0.0;
    double start_time = omp_get_wtime();

    for (iter = 0; iter < max_iterations; iter++) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                u_new[i][j] = coeff * (
                    (u[i-1][j] + u[i+1][j]) / dx2 +
                    (u[i][j-1] + u[i][j+1]) / dy2 -
                    f[i][j]
                );
            }
        }

        // Compute error
        error = 0.0;
        #pragma omp parallel for collapse(2) reduction(max:error)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                double diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > error) error = diff;
            }
        }

        // Copy u_new to u
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                u[i][j] = u_new[i][j];
            }
        }

        if (error < tolerance) {
            iter++;
            break;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    int interior_points = (nx - 2) * (ny - 2);
    long long flops_per_iter = interior_points * 10;
    long long total_flops = flops_per_iter * iter;
    double gflops = (total_flops / elapsed_time) / 1.0e9;

    long long bytes_per_iter = interior_points * (10) * sizeof(double);
    long long total_bytes = bytes_per_iter * iter;
    double mem_bandwidth_gb = (total_bytes / elapsed_time) / 1.0e9;
    double arithmetic_intensity = (double)total_flops / (double)total_bytes;

    fprintf(data_file, "%d,%d,%d,%lld,%.6f,%.6f,%.6f,%.6f\n",
            nx, ny, iter, total_flops, elapsed_time, gflops,
            arithmetic_intensity, mem_bandwidth_gb);

    printf("Converged in %d iterations, time = %.6f s, GFLOPS = %.4f\n\n",
           iter, elapsed_time, gflops);

    for (int i = 0; i < nx; i++) {
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

int main() {
    FILE *data_file = fopen("performance_data.csv", "w");
    if (data_file == NULL) {
        fprintf(stderr, "Error opening performance_data.csv\n");
        return 1;
    }

    fprintf(data_file, "nx,ny,iters,FLOPS,Time,GFLOPS,AI,MemBW\n");

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int nx = sizes[s];
        int ny = sizes[s];
        run_solver(nx, ny, data_file);
    }

    fclose(data_file);
    printf("All results written to performance_data.csv\n");
    return 0;
}
