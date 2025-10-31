#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

// Function to compute the exact solution
double exact_solution(double x, double y) {
    return sin(2.0 * PI * x) * cos(2.0 * PI * y);
}

// Function to compute the source term f(x,y)
double source_term(double x, double y) {
    return -2.0 * (2.0 * PI) * (2.0 * PI) * sin(2.0 * PI * x) * cos(2.0 * PI * y);
}

// Helper function to allocate 2D array
double** allocate_2d_array(int nx, int ny) {
    double **arr = (double **)malloc(nx * sizeof(double *));
    for (int i = 0; i < nx; i++) {
        arr[i] = (double *)malloc(ny * sizeof(double));
    }
    return arr;
}

// Helper function to free 2D array
void free_2d_array(double **arr, int nx) {
    for (int i = 0; i < nx; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Helper function to initialize arrays
void initialize_arrays(double **u, double **u_new, double **f, double **u_exact,
                       int nx, int ny, double x_min, double y_min, 
                       double dx, double dy) {
    // Initialize interior and all points
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = x_min + i * dx;
            double y = y_min + j * dy;
            
            // Initialize solution to zero
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
            
            // Compute source term
            f[i][j] = source_term(x, y);
            
            // Store exact solution for comparison
            u_exact[i][j] = exact_solution(x, y);
        }
    }
    
    // Apply boundary conditions: u = 0 at boundaries
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        u[i][0] = 0.0;
        u[i][ny-1] = 0.0;
        u_new[i][0] = 0.0;
        u_new[i][ny-1] = 0.0;
    }
    
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        u[0][j] = 0.0;
        u[nx-1][j] = 0.0;
        u_new[0][j] = 0.0;
        u_new[nx-1][j] = 0.0;
    }
}

int main(int argc, char *argv[]) {
    // Domain parameters
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    
    // Grid parameters
    int nx = 256;  // Number of points in x-direction
    int ny = 256;  // Number of points in y-direction
    
    if (argc > 1) {
        nx = atoi(argv[1]);
        ny = nx;
    }
    
    double dx = (x_max - x_min) / (nx - 1);
    double dy = (y_max - y_min) / (ny - 1);
    
    // Convergence parameters
    double tolerance = 1.0e-6;
    int max_iterations = 100000;
    
    // Allocate memory for solution arrays
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
            
            // Initialize solution to zero
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
            
            // Compute source term
            f[i][j] = source_term(x, y);
            
            // Store exact solution for comparison
            u_exact[i][j] = exact_solution(x, y);
        }
    }
    
    // Apply boundary conditions: u = 0 at boundaries
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        u[i][0] = 0.0;
        u[i][ny-1] = 0.0;
        u_new[i][0] = 0.0;
        u_new[i][ny-1] = 0.0;
    }
    
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        u[0][j] = 0.0;
        u[nx-1][j] = 0.0;
        u_new[0][j] = 0.0;
        u_new[nx-1][j] = 0.0;
    }
    
    // Precompute coefficients
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coeff = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2));
    
    printf("=== 2D Poisson Equation Solver with OpenMP ===\n");
    printf("Grid size: %d x %d\n", nx, ny);
    printf("Grid spacing: dx = %.6e, dy = %.6e\n", dx, dy);
    printf("Tolerance: %.2e\n", tolerance);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    // Iterative solver using Jacobi method
    int iter;
    double error = 1.0;
    double total_time = 0.0;
    double update_time = 0.0;
    double error_time = 0.0;
    
    printf("Iter\tError\t\tIter Time(s)\tUpdate Time(s)\tError Time(s)\n");
    printf("----\t-----\t\t------------\t--------------\t-------------\n");
    
    for (iter = 0; iter < max_iterations; iter++) {
        double iter_start = omp_get_wtime();
        
        // Update interior points (Jacobi iteration)
        double update_start = omp_get_wtime();
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
        double update_end = omp_get_wtime();
        double update_iter_time = update_end - update_start;
        update_time += update_iter_time;
        
        // Compute error (convergence check)
        double error_start = omp_get_wtime();
        error = 0.0;
        
        #pragma omp parallel for collapse(2) reduction(max:error)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                double diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > error) {
                    error = diff;
                }
            }
        }
        double error_end = omp_get_wtime();
        double error_iter_time = error_end - error_start;
        error_time += error_iter_time;
        
        // Copy u_new to u
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                u[i][j] = u_new[i][j];
            }
        }
        
        double iter_end = omp_get_wtime();
        double iter_time = iter_end - iter_start;
        total_time += iter_time;
        
        // Print progress every 100 iterations or at convergence
        if (iter % 100 == 0 || error < tolerance) {
            printf("%d\t%.6e\t%.6f\t%.6f\t%.6f\n", 
                   iter, error, iter_time, update_iter_time, error_iter_time);
        }
        
        // Check convergence
        if (error < tolerance) {
            iter++;
            break;
        }
    }
    
    printf("\n=== Convergence Results ===\n");
    printf("Converged in %d iterations\n", iter);
    printf("Final error: %.6e\n", error);
    printf("Total time: %.6f seconds\n", total_time);
    printf("Average iteration time: %.6f seconds\n", total_time / iter);
    printf("Total update time: %.6f seconds\n", update_time);
    printf("Total error computation time: %.6f seconds\n", error_time);
    
    // Compute solution error compared to exact solution
    double max_error = 0.0;
    double l2_error = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:max_error) reduction(+:l2_error)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double diff = fabs(u[i][j] - u_exact[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }
            l2_error += diff * diff;
        }
    }
    l2_error = sqrt(l2_error / (nx * ny));
    
    printf("\n=== Accuracy vs Exact Solution ===\n");
    printf("Maximum error: %.6e\n", max_error);
    printf("L2 error: %.6e\n", l2_error);
    
    // Bandwidth and roofline analysis
    printf("\n=== Performance Analysis ===\n");
    
    // Update loop analysis
    long interior_points = (long)(nx - 2) * (ny - 2);
    double updates_per_iter = (double)interior_points;
    
    // Bytes accessed per update:
    // Read: 5 values from u (center + 4 neighbors) + 1 from f = 6 reads
    // Write: 1 value to u_new = 1 write
    // Total: 7 * 8 bytes (double precision) = 56 bytes per point
    long bytes_per_update = 56;
    long total_bytes_update = bytes_per_update * interior_points * iter;
    double bandwidth_update = (total_bytes_update / 1e9) / update_time;  // GB/s
    
    // FLOPs per update:
    // 2 divisions (dx2, dy2), 4 additions, 2 multiplications, 1 subtraction = 9 FLOPs
    long flops_per_update = 9;
    long total_flops_update = flops_per_update * interior_points * iter;
    double gflops_update = (total_flops_update / 1e9) / update_time;
    
    double arithmetic_intensity_update = (double)flops_per_update / bytes_per_update;
    
    printf("Update Loop:\n");
    printf("  Interior points: %ld\n", interior_points);
    printf("  Total iterations: %d\n", iter);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_update);
    printf("  Performance: %.2f GFLOP/s\n", gflops_update);
    printf("  Arithmetic intensity: %.4f FLOP/byte\n", arithmetic_intensity_update);
    
    // Error computation loop analysis
    // Bytes accessed: 2 reads (u_new, u) = 2 * 8 = 16 bytes per point
    long bytes_per_error = 16;
    long total_bytes_error = bytes_per_error * interior_points * iter;
    double bandwidth_error = (total_bytes_error / 1e9) / error_time;  // GB/s
    
    // FLOPs per error check: 1 subtraction, 1 fabs, 1 comparison = 2 FLOPs
    long flops_per_error = 2;
    long total_flops_error = flops_per_error * interior_points * iter;
    double gflops_error = (total_flops_error / 1e9) / error_time;
    
    double arithmetic_intensity_error = (double)flops_per_error / bytes_per_error;
    
    printf("\nError Computation Loop:\n");
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_error);
    printf("  Performance: %.2f GFLOP/s\n", gflops_error);
    printf("  Arithmetic intensity: %.4f FLOP/byte\n", arithmetic_intensity_error);
    
    printf("\n=== Roofline Model Analysis ===\n");
    printf("The update loop has arithmetic intensity of %.4f FLOP/byte\n", arithmetic_intensity_update);
    printf("The error loop has arithmetic intensity of %.4f FLOP/byte\n", arithmetic_intensity_error);
    printf("Both loops are MEMORY-BOUND (typical threshold ~10 FLOP/byte for compute-bound)\n");
    printf("\nTo create roofline plot, compare:\n");
    printf("  - Measured bandwidth vs peak memory bandwidth of your system\n");
    printf("  - Measured GFLOP/s vs peak compute performance\n");
    printf("  - Arithmetic intensity vs ridge point (peak_flops / peak_bandwidth)\n");
    
    // Free memory
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
    
    return 0;
}