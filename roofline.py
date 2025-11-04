import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

PEAK_MEMORY_BW = 280
PEAK_FLOP_RATE = 1.0
RIDGE_POINT = (PEAK_FLOP_RATE * 1000) / PEAK_MEMORY_BW

def load_data(filename):
    """Load CSV and return DataFrame with TFLOPS column"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    
    df = pd.read_csv(filename)
    required_cols = {'nx', 'ny', 'iters', 'FLOPS', 'Time', 'GFLOPS', 'AI', 'MemBW'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"{filename} missing columns: {missing}")
    
    df['TFLOPS'] = df['GFLOPS'] / 1000.0
    df['Label'] = df['nx'].astype(int).astype(str) + '×' + df['ny'].astype(int).astype(str)
    print(f"Loaded {len(df)} points from {filename}")
    return df

def create_roofline_plot(show_labels='selective'):
    fig, ax = plt.subplots(figsize=(16, 10))
    ai_range = np.logspace(-2, 2, 1000)
    memory_bound = ai_range * PEAK_MEMORY_BW / 1000.0
    compute_bound = np.full_like(ai_range, PEAK_FLOP_RATE)
    roofline = np.minimum(memory_bound, compute_bound)

    ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='Theoretical Roofline', zorder=5)
    ax.loglog(RIDGE_POINT, PEAK_FLOP_RATE, 'r*', markersize=20,
              label=f'Ridge Point ({RIDGE_POINT:.3f} FLOP/byte)', zorder=6)

    try:
        df_reg = load_data('performance_data.csv')
        df_mpi = load_data('mpi_performance_data.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    ax.loglog(df_reg['AI'], df_reg['TFLOPS'],
              'o', color='#1f77b4', markersize=10, alpha=0.9,
              markeredgecolor='black', markeredgewidth=0.8,
              label='OpenMP', zorder=4)
    ax.loglog(df_mpi['AI'], df_mpi['TFLOPS'],
              's', color='#d62728', markersize=9, alpha=0.9,
              markeredgecolor='black', markeredgewidth=0.8,
              label='MPI', zorder=4)

    if show_labels in ['all', 'selective']:
        for df, color in [(df_reg, '#1f77b4'), (df_mpi, '#d62728')]:
            for _, row in df.iterrows():
                if show_labels == 'all' or (row['nx'] * row['ny'] >= 100000):
                    ax.annotate(row['Label'],
                                xy=(row['AI'], row['TFLOPS']),
                                xytext=(6, 6), textcoords='offset points',
                                fontsize=7, alpha=0.8,
                                color=color,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2, edgecolor='none'))

    print(f"\n{'='*80}")
    print("ROOFLINE PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Peak Memory BW: {PEAK_MEMORY_BW} GB/s | Peak Compute: {PEAK_FLOP_RATE} TFLOP/s")
    print(f"Ridge Point: {RIDGE_POINT:.3f} FLOP/byte")
    print(f"{'-'*80}")

    for name, df, color in [('Regular', df_reg, '#1f77b4'), ('MPI', df_mpi, '#d62728')]:
        peak_gflops = df['GFLOPS'].max()
        peak_eff = peak_gflops / (PEAK_FLOP_RATE * 1000) * 100
        avg_bw = df['MemBW'].mean()
        bw_eff = avg_bw / PEAK_MEMORY_BW *  100

        print(f"{name:8} | Points: {len(df):2} | "
              f"Peak: {peak_gflops:6.2f} GFLOP/s ({peak_eff:5.1f}%) | "
              f"Avg BW: {avg_bw:6.1f} GB/s ({bw_eff:5.1f}%) | "
              f"AI: {df['AI'].min():.3f}–{df['AI'].max():.3f}")

    print(f"{'-'*80}")
    total_points = len(df_reg) + len(df_mpi)
    print(f"Total data points plotted: {total_points}")
    print(f"{'='*80}\n")

    ax.axvspan(0.01, RIDGE_POINT, alpha=0.1, color='orange', label='Memory Bound', zorder=1)
    ax.axvspan(RIDGE_POINT, 100, alpha=0.1, color='lightblue', label='Compute Bound', zorder=1)
    ax.axhline(y=PEAK_FLOP_RATE, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axvline(x=RIDGE_POINT, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (TFLOP/s)', fontsize=14, fontweight='bold')
    ax.set_title('Roofline Model: Regular vs MPI Poisson Solver', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)

    ax.legend(fontsize=10, loc='lower right', framealpha=0.95,
              edgecolor='black', fancybox=True, shadow=True, ncol=2)

    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.001, 2.0)

    ax.text(0.02, 0.5, 'Memory\nBound', transform=ax.transAxes,
            fontsize=11, color='darkorange', style='italic', ha='left', va='center')
    ax.text(0.85, 0.85, 'Compute\nBound', transform=ax.transAxes,
            fontsize=11, color='steelblue', style='italic', ha='center', va='center')

    plt.tight_layout()
    output_file = 'roofline_regular_vs_mpi.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_file}")

    if '--no-display' not in sys.argv:
        print("Displaying plot... (close window to exit)")
        plt.show()
    else:
        print("Display disabled.")

    plt.close()


if __name__ == "__main__":
    print("Roofline Model: Regular vs MPI Poisson Solver")
    print("="*80)
    print(f"CPU: {PEAK_FLOP_RATE} TFLOP/s, {PEAK_MEMORY_BW} GB/s → Ridge @ {RIDGE_POINT:.3f} FLOP/byte")
    print("="*80)

    label_mode = 'selective'
    if '--label-all' in sys.argv:
        label_mode = 'all'
    elif '--no-labels' in sys.argv:
        label_mode = 'none'

    print(f"Label mode: {label_mode}\n")
    create_roofline_plot(show_labels=label_mode)
    print("\nScript completed!")