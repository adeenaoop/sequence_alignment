# benchmark_alignments.py
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from typing import List, Dict, Callable, Tuple
import pandas as pd

# Import all algorithms (adjust paths as needed)
sys.path.extend(['.', './algorithms'])  # Add your algorithm file paths

# Assuming your files are in the same directory
from smith_waterman import smith_waterman
from needleman_wunsch import needleman_wunsch
from banded_dp import banded_dp
from greedy import greedy_sequence_alignment
from hirschberg import hirschberg

# Wrap algorithms to have consistent interface and capture return values
def run_smith_waterman(seq1, seq2):
    try:
        # Capture print output (the algorithm prints directly)
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            smith_waterman(seq1, seq2)
        return True
    except:
        return False

def run_needleman_wunsch(seq1, seq2):
    try:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            needleman_wunsch(seq1, seq2)
        return True
    except:
        return False

def run_banded_dp(seq1, seq2, k=3):
    try:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            banded_dp(seq1, seq2, k=k)
        return True
    except:
        return False

def run_greedy(seq1, seq2):
    try:
        greedy_sequence_alignment(seq1, seq2)
        return True
    except:
        return False

def run_hirschberg(seq1, seq2):
    try:
        result = hirschberg(seq1, seq2)
        return True
    except:
        return False


class AlignmentBenchmark:
    def __init__(self):
        self.results = {}
        
    def generate_random_sequences(self, length: int, alphabet: str = "ACGT") -> Tuple[str, str]:
        """Generate random DNA sequences for testing."""
        np.random.seed(42)  # For reproducibility
        seq1 = ''.join(np.random.choice(list(alphabet), length))
        
        # Create second sequence with some mutations and indels
        seq2 = list(seq1)
        # Random mutations (10% of positions)
        for i in range(length):
            if np.random.random() < 0.1:
                seq2[i] = np.random.choice(list(alphabet))
        
        # Random insertions/deletions (10% chance)
        if np.random.random() < 0.05:
            pos = np.random.randint(0, length)
            seq2.insert(pos, np.random.choice(list(alphabet)))
        if np.random.random() < 0.05 and len(seq2) > 1:
            pos = np.random.randint(0, len(seq2))
            seq2.pop(pos)
            
        seq2 = ''.join(seq2)
        return seq1, seq2
    
    def measure_memory(self, func: Callable, seq1: str, seq2: str, **kwargs) -> float:
        """Measure peak memory usage of a function."""
        tracemalloc.start()
        try:
            if kwargs:
                func(seq1, seq2, **kwargs)
            else:
                func(seq1, seq2)
        except Exception as e:
            tracemalloc.stop()
            return float('inf')
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024  # Return in KB
    
    def measure_runtime(self, func: Callable, seq1: str, seq2: str, **kwargs) -> float:
        """Measure runtime of a function."""
        start_time = time.perf_counter()
        try:
            if kwargs:
                func(seq1, seq2, **kwargs)
            else:
                func(seq1, seq2)
        except Exception as e:
            return float('inf')
        
        end_time = time.perf_counter()
        return end_time - start_time
    
    def benchmark_algorithms(self, 
                            lengths: List[int], 
                            n_trials: int = 3,
                            verbose: bool = True):
        """Benchmark all algorithms across different sequence lengths."""
        
        algorithms = {
            'Needleman-Wunsch': run_needleman_wunsch,
            'Hirschberg': run_hirschberg,
            'Smith-Waterman': run_smith_waterman,
            'Greedy': run_greedy,
            'Banded DP (k=5)': lambda s1, s2: run_banded_dp(s1, s2, k=5),
            'Banded DP (k=10)': lambda s1, s2: run_banded_dp(s1, s2, k=10),
        }
        
        results = {name: {'times': [], 'memories': [], 'lengths': []} 
                  for name in algorithms.keys()}
        
        for length in lengths:
            if verbose:
                print(f"\nBenchmarking length: {length}")
            
            # Generate sequences once per length for consistency
            seq1, seq2 = self.generate_random_sequences(length)
            
            for algo_name, algo_func in algorithms.items():
                if verbose:
                    print(f"  Testing {algo_name}...")
                
                times = []
                memories = []
                
                for trial in range(n_trials):
                    # Measure runtime
                    runtime = self.measure_runtime(algo_func, seq1, seq2)
                    if runtime != float('inf'):
                        times.append(runtime)
                    
                    # Measure memory
                    memory = self.measure_memory(algo_func, seq1, seq2)
                    if memory != float('inf'):
                        memories.append(memory)
                
                if times:
                    results[algo_name]['times'].append(np.mean(times))
                    results[algo_name]['memories'].append(np.mean(memories))
                else:
                    results[algo_name]['times'].append(float('nan'))
                    results[algo_name]['memories'].append(float('nan'))
                results[algo_name]['lengths'].append(length)
        
        self.results = results
        return results
    
    def fit_complexity(self, lengths: np.ndarray, times: np.ndarray, 
                      complexity: str = 'quadratic') -> Tuple[float, float]:
        """Fit empirical runtime to theoretical complexity."""
        # Remove NaN values
        mask = ~np.isnan(times)
        lengths = lengths[mask]
        times = times[mask]
        
        if len(lengths) < 2:
            return 0, 0
        
        if complexity == 'linear':
            # y = a * x
            X = lengths.reshape(-1, 1)
        elif complexity == 'quadratic':
            # y = a * x^2
            X = (lengths ** 2).reshape(-1, 1)
        elif complexity == 'cubic':
            # y = a * x^3
            X = (lengths ** 3).reshape(-1, 1)
        else:
            X = lengths.reshape(-1, 1)
        
        # Linear regression through origin (or with intercept)
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), times)
        return slope, r_value ** 2  # Return slope and R-squared
    
    def plot_runtime_comparison(self, save_path: str = 'runtime_comparison.png'):
        """Plot runtime comparison with complexity fits."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Algorithm Runtime Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: All algorithms on linear scale
        ax1 = axes[0, 0]
        for algo_name, data in self.results.items():
            if not np.all(np.isnan(data['times'])):
                ax1.plot(data['lengths'], data['times'], 'o-', label=algo_name, linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length (n)', fontsize=12)
        ax1.set_ylabel('Runtime (seconds)', fontsize=12)
        ax1.set_title('Runtime vs Sequence Length', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-log scale to see complexity slopes
        ax2 = axes[0, 1]
        for algo_name, data in self.results.items():
            times = np.array(data['times'])
            lengths = np.array(data['lengths'])
            mask = ~np.isnan(times)
            if np.any(mask):
                ax2.loglog(lengths[mask], times[mask], 'o-', label=algo_name, linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length (n) - log scale', fontsize=12)
        ax2.set_ylabel('Runtime (seconds) - log scale', fontsize=12)
        ax2.set_title('Log-Log Plot (slope = empirical complexity)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Zoom to show smaller algorithms
        ax3 = axes[1, 0]
        fast_algorithms = ['Hirschberg', 'Greedy', 'Banded DP (k=5)', 'Banded DP (k=10)']
        for algo_name in fast_algorithms:
            if algo_name in self.results:
                data = self.results[algo_name]
                if not np.all(np.isnan(data['times'])):
                    ax3.plot(data['lengths'], data['times'], 'o-', label=algo_name, linewidth=2, markersize=8)
        ax3.set_xlabel('Sequence Length (n)', fontsize=12)
        ax3.set_ylabel('Runtime (seconds)', fontsize=12)
        ax3.set_title('Fast Algorithms Comparison', fontsize=12)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Complexity fit demonstration
        ax4 = axes[1, 1]
        nw_data = self.results.get('Needleman-Wunsch', {})
        if nw_data and not np.all(np.isnan(nw_data['times'])):
            lengths = np.array(nw_data['lengths'])
            times = np.array(nw_data['times'])
            mask = ~np.isnan(times)
            lengths = lengths[mask]
            times = times[mask]
            
            ax4.plot(lengths, times, 'bo', label='Empirical', markersize=8)
            
            # Fit quadratic
            coeffs = np.polyfit(lengths, times, 2)
            poly_fit = np.poly1d(coeffs)
            x_smooth = np.linspace(min(lengths), max(lengths), 100)
            ax4.plot(x_smooth, poly_fit(x_smooth), 'r-', label=f'Quadratic fit: {coeffs[0]:.2e}·n²', linewidth=2)
            
            ax4.set_xlabel('Sequence Length (n)', fontsize=12)
            ax4.set_ylabel('Runtime (seconds)', fontsize=12)
            ax4.set_title('Needleman-Wunsch: Empirical vs Quadratic', fontsize=12)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Runtime plot saved to {save_path}")
    
    def plot_memory_comparison(self, save_path: str = 'memory_comparison.png'):
        """Plot memory usage comparison, highlighting Hirschberg's efficiency."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Memory Usage Comparison - Hirschberg\'s Linear Space Advantage', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: All algorithms
        ax1 = axes[0]
        for algo_name, data in self.results.items():
            if not np.all(np.isnan(data['memories'])):
                ax1.plot(data['lengths'], data['memories'], 'o-', label=algo_name, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Sequence Length (n)', fontsize=12)
        ax1.set_ylabel('Peak Memory Usage (KB)', fontsize=12)
        ax1.set_title('Memory Usage vs Sequence Length', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log scale to show stark difference
        ax2 = axes[1]
        for algo_name, data in self.results.items():
            memories = np.array(data['memories'])
            lengths = np.array(data['lengths'])
            mask = ~np.isnan(memories)
            if np.any(mask):
                ax2.semilogy(lengths[mask], memories[mask], 'o-', label=algo_name, linewidth=2, markersize=8)
        
        # Add reference lines
        max_len = max([max(data['lengths']) for data in self.results.values() if data['lengths']])
        x_ref = np.array([1, max_len])
        # O(n) reference
        ax2.plot(x_ref, x_ref * 1, 'k--', alpha=0.5, label='O(n) reference', linewidth=1)
        # O(n²) reference
        ax2.plot(x_ref, (x_ref ** 2) * 0.1, 'k:', alpha=0.5, label='O(n²) reference', linewidth=1)
        
        ax2.set_xlabel('Sequence Length (n)', fontsize=12)
        ax2.set_ylabel('Peak Memory Usage (KB) - log scale', fontsize=12)
        ax2.set_title('Memory Usage (Log Scale) - Hirschberg is O(n)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Memory plot saved to {save_path}")
    
    def print_summary_table(self):
        """Print a summary table of results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Create DataFrame for runtime
        df_runtime = pd.DataFrame()
        df_memory = pd.DataFrame()
        
        for algo_name, data in self.results.items():
            df_runtime[algo_name] = pd.Series(data['times'], index=data['lengths'])
            df_memory[algo_name] = pd.Series(data['memories'], index=data['lengths'])
        
        print("\nRuntime (seconds):")
        print(df_runtime.round(4).to_string())
        print("\nMemory Usage (KB):")
        print(df_memory.round(2).to_string())
        
        # Fit complexities
        print("\n" + "="*80)
        print("COMPLEXITY FITTING (R² values)")
        print("="*80)
        
        for algo_name, data in self.results.items():
            if not np.all(np.isnan(data['times'])):
                lengths = np.array(data['lengths'])
                times = np.array(data['times'])
                
                # Fit different complexities
                linear_r2 = self.fit_complexity(lengths, times, 'linear')[1]
                quad_r2 = self.fit_complexity(lengths, times, 'quadratic')[1]
                
                print(f"\n{algo_name}:")
                print(f"  Linear fit R²:    {linear_r2:.4f}")
                print(f"  Quadratic fit R²: {quad_r2:.4f}")
                print(f"  Best fit: {'Quadratic' if quad_r2 > linear_r2 else 'Linear'}")


# Run benchmarks
def run_full_benchmark():
    """Execute complete benchmark suite."""
    benchmark = AlignmentBenchmark()
    
    # Define sequence lengths to test
    # Start small and increase gradually
    lengths = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
    
    print("Starting benchmark...")
    print(f"Testing sequence lengths: {lengths}")
    print("Note: Some algorithms may fail for larger lengths due to memory/time constraints\n")
    
    # Run benchmarks
    results = benchmark.benchmark_algorithms(lengths, n_trials=3, verbose=True)
    
    # Generate plots
    benchmark.plot_runtime_comparison()
    benchmark.plot_memory_comparison()
    
    # Print summary
    benchmark.print_summary_table()
    
    return benchmark


# Separate script for scaling test (larger n for memory)
def run_memory_scaling_test():
    """Specialized test to clearly show Hirschberg's O(n) memory advantage."""
    benchmark = AlignmentBenchmark()
    
    # Larger lengths for memory test
    lengths = [50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000]
    
    print("\n" + "="*60)
    print("MEMORY SCALING TEST - Larger n to show Hirschberg advantage")
    print("="*60)
    
    # Only test memory-efficient algorithms for large n
    algorithms_to_test = ['Hirschberg', 'Greedy', 'Banded DP (k=10)']
    
    results = {'Hirschberg': {'memories': [], 'lengths': []},
               'Greedy': {'memories': [], 'lengths': []},
               'Banded DP (k=10)': {'memories': [], 'lengths': []}}
    
    for length in lengths:
        print(f"\nTesting length: {length}")
        seq1, seq2 = benchmark.generate_random_sequences(length)
        
        # Test Hirschberg
        memory = benchmark.measure_memory(run_hirschberg, seq1, seq2)
        results['Hirschberg']['memories'].append(memory)
        results['Hirschberg']['lengths'].append(length)
        print(f"  Hirschberg: {memory:.2f} KB")
        
        # Test Greedy
        memory = benchmark.measure_memory(run_greedy, seq1, seq2)
        results['Greedy']['memories'].append(memory)
        results['Greedy']['lengths'].append(length)
        print(f"  Greedy: {memory:.2f} KB")
        
        # Test Banded DP
        memory = benchmark.measure_memory(lambda s1,s2: run_banded_dp(s1,s2,k=10), seq1, seq2)
        results['Banded DP (k=10)']['memories'].append(memory)
        results['Banded DP (k=10)']['lengths'].append(length)
        print(f"  Banded DP (k=10): {memory:.2f} KB")
    
    # Plot memory scaling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo_name, data in results.items():
        if data['memories']:
            ax.plot(data['lengths'], data['memories'], 'o-', label=algo_name, linewidth=2, markersize=8)
    
    # Add theoretical reference lines
    max_len = max(lengths)
    x_ref = np.array([1, max_len])
    ax.plot(x_ref, x_ref * 0.5, 'k--', alpha=0.5, label='O(n) reference', linewidth=2)
    ax.plot(x_ref, (x_ref ** 2) * 0.002, 'k:', alpha=0.5, label='O(n²) reference', linewidth=2)
    
    ax.set_xlabel('Sequence Length (n)', fontsize=12)
    ax.set_ylabel('Peak Memory Usage (KB)', fontsize=12)
    ax.set_title('Memory Scaling: Hirschberg vs Others (Linear Space Advantage)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_scaling_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("MEMORY SCALING SUMMARY")
    print("="*60)
    print("\nAt n=2000:")
    for algo_name, data in results.items():
        if data['lengths'] and len(data['memories']) > 0:
            mem_at_max = data['memories'][-1]
            print(f"  {algo_name}: {mem_at_max:.2f} KB")
    
    return results


if __name__ == "__main__":
    # Run standard benchmark
    benchmark_results = run_full_benchmark()
    
    # Run specialized memory scaling test
    memory_results = run_memory_scaling_test()
