import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# CPU specifications
PEAK_MEMORY_BW = 160  # GB/s
PEAK_FLOP_RATE = 1.0  # TFLOP/s (1000 GFLOP/s)
RIDGE_POINT = (PEAK_FLOP_RATE * 1000) / PEAK_MEMORY_BW  # ops/byte = GFLOP/s / GB/s

def create_roofline_plot(show_labels='selective'):
    """
    Create roofline plot for 2D Poisson solver performance

    Args:
        show_labels: 'all', 'selective', or 'none'
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # Theoretical roofline
    ai_range = np.logspace(-2, 2, 1000)
    memory_bound = ai_range * PEAK_MEMORY_BW / 1000.0  # Convert GB/s to TFLOP/s
    compute_bound = np.full_like(ai_range, PEAK_FLOP_RATE)
    roofline = np.minimum(memory_bound, compute_bound)

    ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='Theoretical Roofline', zorder=5)

    # Mark ridge point
    ax.loglog(RIDGE_POINT, PEAK_FLOP_RATE, 'r*', markersize=20,
              label=f'Ridge Point ({RIDGE_POINT:.2f} ops/byte)', zorder=6)

    # Load and plot performance data
    try:
        df = pd.read_csv('performance_data.csv')
        required_cols = {'nx', 'ny', 'iters', 'FLOPS', 'Time', 'GFLOPS', 'AI', 'MemBW'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

        print(f"✓ Loaded {len(df)} data points")

        # Convert GFLOPS to TFLOPS
        df['TFLOPS'] = df['GFLOPS'] / 1000.0

        ax.loglog(df['AI'], df['TFLOPS'],
                  'o',
                  color='#d62728',
                  markersize=10,
                  alpha=0.8,
                  label='Measured Performance',
                  markeredgecolor='black',
                  markeredgewidth=0.8,
                  zorder=4)

        # Optionally show labels
        if show_labels in ['all', 'selective']:
            for _, row in df.iterrows():
                if show_labels == 'all' or row['nx'] * row['ny'] >= 100000:
                    label = f"{int(row['nx'])}×{int(row['ny'])}"
                    ax.annotate(label,
                                xy=(row['AI'], row['TFLOPS']),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=7, alpha=0.7,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='#d62728',
                                          alpha=0.2, edgecolor='none'))

        # Print performance summary
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Test cases: {len(df)}")
        print(f"Peak achieved: {df['GFLOPS'].max():.3f} GFLOP/s "
              f"({df['GFLOPS'].max()/(PEAK_FLOP_RATE*1000)*100:.2f}% of peak)")
        print(f"Average: {df['GFLOPS'].mean():.3f} GFLOP/s")
        print(f"Avg Memory BW: {df['MemBW'].mean():.2f} GB/s "
              f"({df['MemBW'].mean()/PEAK_MEMORY_BW*100:.1f}% of peak)")
        print(f"AI range: {df['AI'].min():.3f} - {df['AI'].max():.3f} FLOP/byte")

        memory_bound = df[df['AI'] < RIDGE_POINT]
        compute_bound = df[df['AI'] >= RIDGE_POINT]
        print(f"\nMemory-bound cases: {len(memory_bound)}/{len(df)} "
              f"({len(memory_bound)/len(df)*100:.1f}%)")
        print(f"Compute-bound cases: {len(compute_bound)}/{len(df)} "
              f"({len(compute_bound)/len(df)*100:.1f}%)")
        print(f"{'='*80}\n")

    except FileNotFoundError:
        print("⚠️ Could not find performance_data.csv")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Shade regions
    ax.axvspan(0.01, RIDGE_POINT, alpha=0.1, color='orange', label='Memory Bound', zorder=1)
    ax.axvspan(RIDGE_POINT, 100, alpha=0.1, color='lightblue', label='Compute Bound', zorder=1)

    # Reference lines
    ax.axhline(y=PEAK_FLOP_RATE, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=RIDGE_POINT, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (TFLOP/s)', fontsize=14, fontweight='bold')
    ax.set_title('Roofline Model: 2D Poisson Solver Performance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)

    ax.legend(fontsize=9, loc='lower right', framealpha=0.95,
              edgecolor='black', fancybox=True, shadow=True, ncol=2)

    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.001, 2.0)

    ax.text(0.03, 0.5, 'Memory\nBound', transform=ax.transAxes,
            fontsize=11, alpha=0.6, style='italic', ha='left')
    ax.text(0.85, 0.85, 'Compute\nBound', transform=ax.transAxes,
            fontsize=11, alpha=0.6, style='italic', ha='center')

    plt.tight_layout()

    # Save figure
    output_file = 'roofline_model_poisson.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")

    if '--no-display' not in sys.argv:
        print("Displaying plot (close window to continue)...")
        plt.show()
    else:
        print("Display disabled (--no-display)")

    plt.close()


if __name__ == "__main__":
    print("Roofline Model Analysis: 2D Poisson Solver")
    print("="*80)
    print(f"CPU Specifications:")
    print(f"  Peak Memory Bandwidth: {PEAK_MEMORY_BW} GB/s")
    print(f"  Peak FLOP Rate: {PEAK_FLOP_RATE} TFLOP/s")
    print(f"  Ridge Point: {RIDGE_POINT:.3f} ops/byte")
    print("="*80 + "\n")

    label_mode = 'selective'
    if '--label-all' in sys.argv:
        label_mode = 'all'
    elif '--no-labels' in sys.argv:
        label_mode = 'none'

    print(f"Label mode: {label_mode}\n")
    create_roofline_plot(show_labels=label_mode)
    print("Script completed successfully!")
