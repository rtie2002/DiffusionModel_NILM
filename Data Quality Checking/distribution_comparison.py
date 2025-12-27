"""
Data Distribution Comparison Tool
数据分布对比工具

Industry-standard methods for comparing real vs synthetic data distributions:
1. Statistical tests (KS test, Chi-square test)
2. Visual comparison (histograms, Q-Q plots, CDFs)
3. Distribution metrics (mean, std, quantiles)
4. Correlation analysis

用于验证合成数据是否与真实数据具有相同的分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tkinter import Tk, filedialog
import sys

class DistributionComparator:
    def __init__(self, real_path, synthetic_path=None):
        self.real_path = real_path
        self.synthetic_path = synthetic_path
        
        # Load data
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        self.load_data()
        
        # Run comparison
        if self.synthetic_path:
            self.compare_distributions()
        else:
            self.analyze_single_distribution()
    
    def load_data(self):
        """Load real and synthetic data"""
        # Load real data
        print(f"\nLoading real data: {self.real_path}")
        
        # Auto-detect number of columns
        df_temp = pd.read_csv(self.real_path, header=None, nrows=1)
        num_cols = len(df_temp.columns)
        
        if num_cols == 5:
            # 5-column format: appliance, minute, hour, day, month
            self.df_real = pd.read_csv(
                self.real_path,
                header=None,
                names=['appliance', 'minute', 'hour', 'day', 'month']
            )
            print(f"✓ Detected 5-column format (appliance, minute, hour, day, month)")
        elif num_cols == 6:
            # 6-column format: aggregate, appliance, minute, hour, day, month
            self.df_real = pd.read_csv(
                self.real_path,
                header=None,
                names=['aggregate', 'appliance', 'minute', 'hour', 'day', 'month']
            )
            print(f"✓ Detected 6-column format (aggregate, appliance, minute, hour, day, month)")
        else:
            raise ValueError(f"Unexpected number of columns: {num_cols}. Expected 5 or 6.")
        
        # Drop rows with NaN values
        before_len = len(self.df_real)
        self.df_real = self.df_real.dropna()
        after_len = len(self.df_real)
        if before_len != after_len:
            print(f"⚠ Dropped {before_len - after_len} rows with NaN values")
        
        print(f"✓ Real data: {len(self.df_real):,} rows")
        
        # Load synthetic data if provided
        if self.synthetic_path:
            print(f"\nLoading synthetic data: {self.synthetic_path}")
            
            # Auto-detect for synthetic too
            df_temp = pd.read_csv(self.synthetic_path, header=None, nrows=1)
            num_cols = len(df_temp.columns)
            
            if num_cols == 5:
                self.df_synthetic = pd.read_csv(
                    self.synthetic_path,
                    header=None,
                    names=['appliance', 'minute', 'hour', 'day', 'month']
                )
            elif num_cols == 6:
                self.df_synthetic = pd.read_csv(
                    self.synthetic_path,
                    header=None,
                    names=['aggregate', 'appliance', 'minute', 'hour', 'day', 'month']
                )
            else:
                raise ValueError(f"Unexpected number of columns: {num_cols}. Expected 5 or 6.")
            
            # Drop NaN
            before_len = len(self.df_synthetic)
            self.df_synthetic = self.df_synthetic.dropna()
            after_len = len(self.df_synthetic)
            if before_len != after_len:
                print(f"⚠ Dropped {before_len - after_len} rows with NaN values")
            
            print(f"✓ Synthetic data: {len(self.df_synthetic):,} rows")
    
    def kolmogorov_smirnov_test(self, real_data, synthetic_data, feature_name):
        """
        Kolmogorov-Smirnov test - Industry standard for distribution comparison
        H0: Two samples come from the same distribution
        """
        statistic, p_value = stats.ks_2samp(real_data, synthetic_data)
        
        # Interpretation
        alpha = 0.05
        same_distribution = p_value > alpha
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'same_distribution': same_distribution,
            'interpretation': 'PASS' if same_distribution else 'FAIL'
        }
    
    def chi_square_test(self, real_data, synthetic_data, bins=50):
        """
        Chi-square test for categorical/binned data
        """
        # Create histograms
        real_hist, bin_edges = np.histogram(real_data, bins=bins)
        synthetic_hist, _ = np.histogram(synthetic_data, bins=bin_edges)
        
        # Avoid division by zero
        real_hist = real_hist + 1
        synthetic_hist = synthetic_hist + 1
        
        # Chi-square test
        statistic, p_value = stats.chisquare(synthetic_hist, real_hist)
        
        alpha = 0.05
        same_distribution = p_value > alpha
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'same_distribution': same_distribution,
            'interpretation': 'PASS' if same_distribution else 'FAIL'
        }
    
    def calculate_distribution_metrics(self, data, name):
        """Calculate comprehensive distribution metrics"""
        return {
            'name': name,
            'count': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'q25': np.percentile(data, 25),
            'median': np.median(data),
            'q75': np.percentile(data, 75),
            'max': np.max(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    def compare_distributions(self):
        """Compare real vs synthetic distributions"""
        print("\n" + "="*80)
        print("DISTRIBUTION COMPARISON ANALYSIS")
        print("="*80)
        
        features = ['aggregate', 'appliance', 'minute', 'hour', 'day', 'month']
        results = {}
        
        for feature in features:
            print(f"\n{'='*80}")
            print(f"Feature: {feature.upper()}")
            print(f"{'='*80}")
            
            real_data = self.df_real[feature].values
            synthetic_data = self.df_synthetic[feature].values
            
            # 1. Statistical Tests
            print("\n1. STATISTICAL TESTS")
            print("-" * 80)
            
            # KS Test
            ks_result = self.kolmogorov_smirnov_test(real_data, synthetic_data, feature)
            print(f"\nKolmogorov-Smirnov Test:")
            print(f"  Statistic: {ks_result['statistic']:.6f}")
            print(f"  P-value: {ks_result['p_value']:.6f}")
            print(f"  Result: {ks_result['interpretation']} (p > 0.05 means same distribution)")
            
            # Chi-square Test (for discrete features)
            if feature in ['minute', 'hour', 'day', 'month']:
                chi_result = self.chi_square_test(real_data, synthetic_data)
                print(f"\nChi-Square Test:")
                print(f"  Statistic: {chi_result['statistic']:.6f}")
                print(f"  P-value: {chi_result['p_value']:.6f}")
                print(f"  Result: {chi_result['interpretation']}")
            
            # 2. Distribution Metrics
            print("\n2. DISTRIBUTION METRICS")
            print("-" * 80)
            
            real_metrics = self.calculate_distribution_metrics(real_data, 'Real')
            synthetic_metrics = self.calculate_distribution_metrics(synthetic_data, 'Synthetic')
            
            print(f"\n{'Metric':<15} {'Real':<15} {'Synthetic':<15} {'Diff %':<15}")
            print("-" * 60)
            
            for key in ['mean', 'std', 'min', 'q25', 'median', 'q75', 'max']:
                real_val = real_metrics[key]
                synth_val = synthetic_metrics[key]
                diff_pct = ((synth_val - real_val) / real_val * 100) if real_val != 0 else 0
                print(f"{key:<15} {real_val:<15.4f} {synth_val:<15.4f} {diff_pct:<15.2f}")
            
            # Store results
            results[feature] = {
                'ks_test': ks_result,
                'real_metrics': real_metrics,
                'synthetic_metrics': synthetic_metrics
            }
        
        # Summary
        self.print_summary(results)
        
        # Visualize
        self.visualize_comparison(results)
        
        return results
    
    def analyze_single_distribution(self):
        """Analyze single dataset distribution"""
        print("\n" + "="*80)
        print("SINGLE DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Only analyze columns that exist
        features = [col for col in ['aggregate', 'appliance', 'minute', 'hour', 'day', 'month'] 
                   if col in self.df_real.columns]
        
        for feature in features:
            print(f"\n{feature.upper()}:")
            data = self.df_real[feature].values
            metrics = self.calculate_distribution_metrics(data, 'Real')
            
            print(f"  Count: {metrics['count']:,}")
            print(f"  Mean: {metrics['mean']:.4f}")
            print(f"  Std: {metrics['std']:.4f}")
            print(f"  Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
            print(f"  Quartiles: Q25={metrics['q25']:.4f}, Median={metrics['median']:.4f}, Q75={metrics['q75']:.4f}")
        
        # Visualize
        self.visualize_single_distribution()
    
    def print_summary(self, results):
        """Print overall summary"""
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for feature, result in results.items():
            if result['ks_test']['same_distribution']:
                passed_tests += 1
            total_tests += 1
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nStatistical Tests:")
        print(f"  Total: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 80:
            print(f"\n✓ CONCLUSION: Synthetic data has SIMILAR distribution to real data")
        elif pass_rate >= 50:
            print(f"\n⚠ CONCLUSION: Synthetic data has PARTIALLY SIMILAR distribution to real data")
        else:
            print(f"\n✗ CONCLUSION: Synthetic data has DIFFERENT distribution from real data")
    
    def visualize_comparison(self, results):
        """Visualize distribution comparison"""
        features = list(results.keys())
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Real vs Synthetic Distribution Comparison', fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            real_data = self.df_real[feature].values
            synthetic_data = self.df_synthetic[feature].values
            
            # Determine bins
            if feature in ['minute', 'hour', 'day', 'month']:
                bins = int(real_data.max() - real_data.min() + 1)
            else:
                bins = 50
            
            # Plot histograms
            ax.hist(real_data, bins=bins, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(synthetic_data, bins=bins, alpha=0.5, label='Synthetic', color='red', density=True)
            
            # Add KS test result
            ks_result = results[feature]['ks_test']
            status = '✓' if ks_result['same_distribution'] else '✗'
            ax.set_title(f"{feature} {status} (p={ks_result['p_value']:.4f})", fontsize=10)
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: distribution_comparison.png")
        plt.show()
    
    def visualize_single_distribution(self):
        """Visualize single dataset distribution"""
        # Only plot columns that exist
        features = [col for col in ['aggregate', 'appliance', 'minute', 'hour', 'day', 'month'] 
                   if col in self.df_real.columns]
        
        num_features = len(features)
        rows = (num_features + 1) // 2  # Ceiling division
        
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4*rows))
        fig.suptitle('Data Distribution Analysis', fontsize=14, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            data = self.df_real[feature].values
            
            # Determine bins
            if feature in ['minute', 'hour', 'day', 'month']:
                bins = int(data.max() - data.min() + 1)
            else:
                bins = 50
            
            # Plot histogram
            ax.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f"{feature}", fontsize=10)
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean = np.mean(data)
            std = np.std(data)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.2f}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(num_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: distribution_analysis.png")
        plt.show()

def select_file(title="Select CSV File"):
    """Open file dialog"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir="created_data/UK_DALE/"
    )
    root.destroy()
    return file_path

def main():
    print("="*80)
    print("DATA DISTRIBUTION COMPARISON TOOL")
    print("="*80)
    print("\nIndustry-standard methods:")
    print("  1. Kolmogorov-Smirnov Test (KS Test)")
    print("  2. Chi-Square Test")
    print("  3. Distribution Metrics (mean, std, quantiles)")
    print("  4. Visual Comparison (histograms)")
    
    # Select files
    print("\n" + "="*80)
    print("FILE SELECTION")
    print("="*80)
    
    if len(sys.argv) > 1:
        real_path = sys.argv[1]
        synthetic_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("\nSelect REAL data file:")
        real_path = select_file("Select Real Data CSV")
        
        if not real_path:
            print("\n✗ No file selected. Exiting...")
            return
        
        choice = input("\nDo you want to compare with SYNTHETIC data? (y/n): ").strip().lower()
        if choice == 'y':
            print("\nSelect SYNTHETIC data file:")
            synthetic_path = select_file("Select Synthetic Data CSV")
        else:
            synthetic_path = None
    
    # Run comparison
    try:
        comparator = DistributionComparator(real_path, synthetic_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
