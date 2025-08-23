#!/usr/bin/env python3
"""
Simple validation script for Monte Carlo simulation
Testing the core logic with minimal iterations
"""

import numpy as np
import sys
import os

# Add the scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import our simulation class
from monte_carlo_simulation import TranslationTimelineMonteCarlo

def quick_validation_test():
    """Quick test to validate our simulation produces reasonable results"""
    print("🔬 QUICK VALIDATION TEST")
    print("=" * 50)
    
    simulator = TranslationTimelineMonteCarlo()
    
    # Test individual model components
    print("Testing model components:")
    
    # Current time (August 2025)
    current_t = 2025 - 2020  # 5.0 years from baseline
    current_quality = simulator.logistic_growth_model(current_t)
    print(f"   Current quality (Aug 2025): {current_quality:.3f} ({current_quality*100:.1f}%)")
    
    # Test quality in 1 year (2026)
    future_quality = simulator.logistic_growth_model(current_t + 1)
    print(f"   Projected quality (Aug 2026): {future_quality:.3f} ({future_quality*100:.1f}%)")
    
    # Expected 90% threshold crossing
    target_quality = 0.90
    for months_ahead in range(1, 25):  # Check next 24 months
        future_t = current_t + (months_ahead / 12.0)
        quality = simulator.logistic_growth_model(future_t)
        if quality >= target_quality:
            print(f"   90% threshold reached in: {months_ahead} months ({simulator.months_to_quarter(months_ahead)})")
            break
    
    # Run small Monte Carlo test
    print("\n🎲 Mini Monte Carlo test (100 iterations):")
    results = simulator.monte_carlo_simulation(n_simulations=100, verbose=False)
    
    threshold_median = results['threshold_time_months'].median()
    displacement_median = results['displacement_time_months'].median()
    
    print(f"   90% performance median: {threshold_median:.1f} months ({simulator.months_to_quarter(threshold_median)})")
    print(f"   Market displacement median: {displacement_median:.1f} months ({simulator.months_to_quarter(displacement_median)})")
    
    # Compare with expected results from paper
    expected_threshold = 13.2  # Q4 2025 from paper
    expected_displacement = 21.1  # Q2 2026 from paper
    
    print(f"\n📋 Comparison with paper results:")
    print(f"   Expected 90% performance: {expected_threshold} months (Q4 2025)")
    print(f"   Our simulation: {threshold_median:.1f} months ({simulator.months_to_quarter(threshold_median)})")
    print(f"   Difference: {abs(threshold_median - expected_threshold):.1f} months")
    
    print(f"   Expected market displacement: {expected_displacement} months (Q2 2026)")
    print(f"   Our simulation: {displacement_median:.1f} months ({simulator.months_to_quarter(displacement_median)})")
    print(f"   Difference: {abs(displacement_median - expected_displacement):.1f} months")
    
    # Validation check
    threshold_close = abs(threshold_median - expected_threshold) < 6  # Within 6 months
    displacement_close = abs(displacement_median - expected_displacement) < 12  # Within 1 year
    
    if threshold_close and displacement_close:
        print(f"\n✅ VALIDATION PASSED: Results are reasonably close to expected values")
    else:
        print(f"\n⚠️  VALIDATION WARNING: Results differ significantly from expected values")
        print(f"   This may indicate model parameter adjustment is needed")
    
    return results

if __name__ == "__main__":
    results = quick_validation_test()
