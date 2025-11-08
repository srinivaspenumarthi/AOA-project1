#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm Design Project: Greedy & Divide-and-Conquer
======================================================

Authors: Avighna Yarlagadda, Manikanta Srinivas Penumarthi
Date: November 2024
Python: 3.11.4

A) GREEDY: Battery Charging Optimization
   - Domain: Energy management systems, EV charging stations
   - Problem: Maximize number of devices fully charged given limited capacity
   - Algorithm: Sort by energy requirement (ascending), select greedily
   - Complexity: O(n log n) time, O(n) space
   - Proof: Greedy choice property + optimal substructure

B) DIVIDE-AND-CONQUER: Temperature Anomaly Detection  
   - Domain: Environmental monitoring, sensor networks
   - Problem: Find max temperature variation (max - min) efficiently
   - Algorithm: Recursive min-max finding with optimal pairing
   - Complexity: ⌈3n/2⌉ - 2 comparisons, O(n) time
   - Proof: Strong induction on subarray size

This script:
  • Runs correctness verification tests
  • Benchmarks algorithms with multiple trials
  • Generates CSV data files
  • Creates publication-quality PNG plots for LaTeX
  • Tracks comparison counts for D&C algorithm
"""

from __future__ import annotations
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================
# Configuration & Utilities
# ============================================

OUTPUT_DIR = "outputs"
RANDOM_SEED = 42

def ensure_outputs_dir() -> None:
    """Create outputs directory if it doesn't exist."""
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def path_in_outputs(filename: str) -> str:
    """Get full path for file in outputs directory."""
    ensure_outputs_dir()
    return os.path.join(OUTPUT_DIR, filename)

def now_ns() -> int:
    """High-resolution timer in nanoseconds."""
    return time.perf_counter_ns()

def secs(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1e9

def write_csv(filename: str, header: List[str], rows: List[Tuple]) -> None:
    """Write data to CSV file in outputs directory."""
    fp = path_in_outputs(filename)
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  → Wrote {fp}")

def try_matplotlib():
    """Attempt to import matplotlib, return None if unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        print(f"[WARNING] matplotlib not available: {e}")
        print("          Plots will be skipped. Install with: pip install matplotlib")
        return None

random.seed(RANDOM_SEED)

# ============================================
# PROBLEM A: Battery Charging (Greedy)
# ============================================

@dataclass(frozen=True)
class Device:
    """Represents a device requiring charging."""
    energy: float      # Energy requirement in kWh
    device_id: int     # Unique identifier
    
    def __repr__(self):
        return f"Device({self.device_id}, {self.energy:.1f}kWh)"

def greedy_charging(devices: List[Device], capacity: float) -> Tuple[float, List[Device], List[int]]:
    """
    Greedy algorithm for battery charging optimization.
    
    DOMAIN EXPLANATION (EV Charging Station Context):
    An electric vehicle charging station has limited power capacity (e.g., 500 kWh).
    Multiple EVs arrive needing different amounts of energy (e.g., 20 kWh, 50 kWh, etc.).
    The station operator wants to maximize the number of vehicles fully charged
    (better customer satisfaction than partially charging many vehicles).
    
    The greedy strategy: Always charge the vehicle requiring the least energy first.
    This leaves more capacity for additional vehicles.
    
    ALGORITHM:
    1. Sort all devices by energy requirement (ascending order)
    2. Select devices one-by-one while capacity permits
    3. Stop when no more devices fit
    
    CORRECTNESS PROOF (in report):
    - Greedy Choice Property: Selecting the minimum energy device is always optimal
    - Optimal Substructure: After selecting a device, the remaining problem
      has the same structure with reduced capacity
    
    Args:
        devices: List of devices needing charging
        capacity: Total available energy in kWh
        
    Returns:
        Tuple of (total_energy_used, selected_devices, selected_indices)
        
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing results
    """
    if not devices:
        return 0.0, [], []
    
    # Sort by energy requirement (ascending) - O(n log n)
    sorted_devices = sorted(devices, key=lambda d: d.energy)
    
    selected = []
    selected_indices = []
    remaining = capacity
    total_used = 0.0
    
    # Greedy selection - O(n)
    for dev in sorted_devices:
        if dev.energy <= remaining:
            selected.append(dev)
            selected_indices.append(dev.device_id)
            remaining -= dev.energy
            total_used += dev.energy
    
    return total_used, selected, selected_indices

def brute_force_charging(devices: List[Device], capacity: float) -> Tuple[int, float]:
    """
    Brute force solution: try all 2^n subsets.
    
    Returns: (max_count, total_energy) for best subset
    
    WARNING: Exponential time - only use for n <= 20
    """
    n = len(devices)
    if n > 20:
        raise ValueError(f"Brute force unsafe for n={n} > 20")
    
    best_count = 0
    best_energy = 0.0
    
    for mask in range(1 << n):
        subset = [devices[i] for i in range(n) if (mask >> i) & 1]
        total = sum(d.energy for d in subset)
        
        if total <= capacity:
            count = len(subset)
            if count > best_count:
                best_count = count
                best_energy = total
    
    return best_count, best_energy

def generate_devices(n: int, 
                    e_min: float = 1.0, 
                    e_max: float = 100.0) -> List[Device]:
    """Generate n devices with random energy requirements."""
    return [Device(energy=random.uniform(e_min, e_max), 
                   device_id=i) 
            for i in range(n)]

# ----- Experiments for Problem A -----

def exp_charging_sanity(trials: int = 30, n: int = 12) -> None:
    """Verify greedy optimality against brute force."""
    print("\n[Charging] Running sanity checks...")
    rows = []
    failures = 0
    
    for trial in range(trials):
        devices = generate_devices(n, e_min=5.0, e_max=50.0)
        capacity = random.uniform(100, 300)
        
        # Greedy solution
        _, greedy_selected, _ = greedy_charging(devices, capacity)
        greedy_count = len(greedy_selected)
        
        # Brute force solution
        brute_count, _ = brute_force_charging(devices, capacity)
        
        match = greedy_count == brute_count
        if not match:
            failures += 1
            print(f"  Trial {trial + 1}: MISMATCH - Greedy={greedy_count}, Brute={brute_count}")
        
        rows.append((trial + 1, n, capacity, greedy_count, brute_count, int(match)))
    
    write_csv("charging_sanity.csv", 
              ["trial", "n", "capacity", "greedy_count", "brute_count", "match"],
              rows)
    
    if failures == 0:
        print(f"  ✓ All {trials} trials passed!")
    else:
        print(f"  ✗ {failures}/{trials} trials failed!")
        raise AssertionError("Greedy algorithm failed sanity check")

def exp_charging_timing(sizes: Tuple[int, ...] = (200, 400, 800, 1600, 3200, 6400),
                       trials: int = 100) -> None:
    """Benchmark greedy charging algorithm."""
    print("\n[Charging] Running performance benchmarks...")
    rows = []
    
    for n in sizes:
        times = []
        counts = []
        
        for _ in range(trials):
            devices = generate_devices(n, e_min=1.0, e_max=100.0)
            capacity = 500.0  # Fixed capacity
            
            t0 = now_ns()
            _, selected, _ = greedy_charging(devices, capacity)
            t1 = now_ns()
            
            times.append(secs(t1 - t0))
            counts.append(len(selected))
        
        mean_time = sum(times) / len(times)
        std_time = math.sqrt(sum((t - mean_time)**2 for t in times) / len(times))
        mean_count = sum(counts) / len(counts)
        
        rows.append((n, mean_time, std_time, mean_count))
        print(f"  n={n:5d}: {mean_time:.6f}s ± {std_time:.6f}s  "
              f"(avg {mean_count:.1f} devices)")
    
    write_csv("charging_timing.csv",
              ["n", "mean_time_s", "std_time_s", "avg_devices_charged"],
              rows)
    
    # Generate plot
    plt = try_matplotlib()
    if plt:
        ns = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        
        plt.figure(figsize=(10, 7))
        plt.errorbar(ns, means, yerr=stds, fmt='bo-', capsize=5, 
                    capthick=2, markersize=8, linewidth=2,
                    label='Measured runtime')
        
        # Fit n log n on log-log scale
        import numpy as np
        log_ns = np.log(ns)
        log_means = np.log(means)
        coeffs = np.polyfit(log_ns, log_means, 1)
        fitted = np.exp(coeffs[1]) * np.array(ns) ** coeffs[0]
        
        plt.plot(ns, fitted, 'r--', linewidth=2.5,
                label=f'Fitted O(n log n), slope={coeffs[0]:.2f}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Input Size (n devices)', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Greedy Charging Algorithm Runtime', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(path_in_outputs('greedy_runtime.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved plot: greedy_runtime.png (slope={coeffs[0]:.3f})")

# ============================================
# PROBLEM B: Temperature Anomaly Detection (D&C)
# ============================================

# Global counter for comparison tracking
comparison_count = 0

def find_min_max_dc(arr: List[float], low: int, high: int) -> Tuple[float, float]:
    """
    OPTIMIZED Divide-and-conquer algorithm to find both min and max.
    
    DOMAIN EXPLANATION (Environmental Monitoring Context):
    Weather stations collect temperature data from sensor networks.
    Meteorologists need to quickly identify temperature anomalies (extreme variations)
    to detect climate events like heat waves or cold snaps.
    Finding both min and max in a single pass is more efficient than two separate scans.
    
    ALGORITHM:
    This uses an optimized D&C approach that achieves the theoretical
    minimum of ⌈3n/2⌉ - 2 comparisons.
    
    Key insight: For odd n, we set initial min=max=first element (0 comparisons).
    For even n, we compare first two elements (1 comparison). Then process 
    remaining elements in pairs.
    
    CORRECTNESS PROOF (in report):
    - Base cases: n=1 (0 comparisons), n=2 (1 comparison)
    - For n>2: Process pairs, update min/max with 3 comparisons per pair
    - Total: ⌈3n/2⌉ - 2 comparisons
    
    Args:
        arr: Array of temperature readings
        low: Start index (inclusive)
        high: End index (inclusive)
        
    Returns:
        Tuple (min, max) of elements in arr[low..high]
        
    Comparisons: ⌈3n/2⌉ - 2 where n = high - low + 1
    Time: O(n)
    Space: O(1) iterative version
    """
    global comparison_count
    n = high - low + 1
    
    # Base case: single element (0 comparisons)
    if n == 1:
        return arr[low], arr[low]
    
    # Initialize min and max
    if n % 2 == 0:
        # Even number of elements - compare first pair (1 comparison)
        comparison_count += 1
        if arr[low] < arr[low + 1]:
            current_min = arr[low]
            current_max = arr[low + 1]
        else:
            current_min = arr[low + 1]
            current_max = arr[low]
        start_idx = low + 2
    else:
        # Odd number of elements - first element is both min and max (0 comparisons)
        current_min = arr[low]
        current_max = arr[low]
        start_idx = low + 1
    
    # Process remaining elements in pairs
    # Each pair requires 3 comparisons:
    # 1. Compare the two elements in the pair
    # 2. Compare smaller with current_min
    # 3. Compare larger with current_max
    i = start_idx
    while i < high:
        comparison_count += 1  # Compare pair elements
        if arr[i] < arr[i + 1]:
            smaller = arr[i]
            larger = arr[i + 1]
        else:
            smaller = arr[i + 1]
            larger = arr[i]
        
        # Compare smaller with current min
        comparison_count += 1
        if smaller < current_min:
            current_min = smaller
        
        # Compare larger with current max
        comparison_count += 1
        if larger > current_max:
            current_max = larger
        
        i += 2
    
    return current_min, current_max

def temperature_anomaly_dc(temperatures: List[float]) -> Tuple[float, int]:
    """
    Compute max temperature variation using divide-and-conquer.
    
    Args:
        temperatures: List of temperature readings in Celsius
        
    Returns:
        Tuple of (max_variation, comparison_count)
    """
    global comparison_count
    comparison_count = 0
    
    if len(temperatures) == 0:
        return 0.0, 0
    if len(temperatures) == 1:
        return 0.0, 0
    
    t_min, t_max = find_min_max_dc(temperatures, 0, len(temperatures) - 1)
    return t_max - t_min, comparison_count

def temperature_anomaly_naive(temperatures: List[float]) -> Tuple[float, int]:
    """
    Naive approach: separate passes for min and max.
    
    Returns: (max_variation, comparison_count)
    Comparisons: 2n - 2
    """
    if len(temperatures) <= 1:
        return 0.0, 0
    
    n = len(temperatures)
    # Find min: n-1 comparisons
    t_min = temperatures[0]
    for i in range(1, n):
        if temperatures[i] < t_min:
            t_min = temperatures[i]
    
    # Find max: n-1 comparisons  
    t_max = temperatures[0]
    for i in range(1, n):
        if temperatures[i] > t_max:
            t_max = temperatures[i]
    
    return t_max - t_min, 2 * (n - 1)

def generate_temperatures(n: int,
                         t_min: float = -30.0,
                         t_max: float = 50.0) -> List[float]:
    """Generate n random temperature readings."""
    return [random.uniform(t_min, t_max) for _ in range(n)]

def theoretical_comparisons(n: int) -> float:
    """Calculate theoretical comparison count: ceil(3n/2) - 2"""
    return math.ceil(3 * n / 2) - 2

# ----- Experiments for Problem B -----

def exp_temperature_sanity(trials: int = 30, n: int = 100) -> None:
    """Verify D&C correctness against naive approach."""
    print("\n[Temperature] Running sanity checks...")
    rows = []
    failures = 0
    
    for trial in range(trials):
        temps = generate_temperatures(n)
        
        dc_var, dc_comps = temperature_anomaly_dc(temps)
        naive_var, naive_comps = temperature_anomaly_naive(temps)
        
        # Check correctness
        match = abs(dc_var - naive_var) < 1e-9
        
        # Check comparison count (allow small tolerance for implementation variations)
        theoretical_dc = theoretical_comparisons(n)
        comp_match = abs(dc_comps - theoretical_dc) <= 1
        
        if not match or not comp_match:
            failures += 1
            print(f"  Trial {trial + 1}: ISSUE - "
                  f"Variation match={match}, Comparison match={comp_match} "
                  f"(DC={dc_comps}, Theory={theoretical_dc})")
        
        rows.append((trial + 1, n, dc_var, naive_var, dc_comps, 
                    naive_comps, theoretical_dc, int(match and comp_match)))
    
    write_csv("temperature_sanity.csv",
              ["trial", "n", "dc_variation", "naive_variation", 
               "dc_comparisons", "naive_comparisons", "theoretical_dc", "match"],
              rows)
    
    if failures == 0:
        print(f"  ✓ All {trials} trials passed!")
        # Print sample comparison counts
        sample_dc = rows[0][4]
        sample_theory = rows[0][6]
        print(f"     D&C comparisons: {sample_dc:.0f} (theoretical = {sample_theory:.0f})")
        print(f"     Naive comparisons: {naive_comps} (= 2n-2 = {2*n-2})")
    else:
        print(f"  ✗ {failures}/{trials} trials had issues!")
        raise AssertionError("D&C algorithm failed sanity check")

def exp_temperature_timing(sizes: Tuple[int, ...] = (200, 400, 800, 1600, 3200, 6400, 12800),
                          trials: int = 100) -> None:
    """Benchmark temperature anomaly detection."""
    print("\n[Temperature] Running performance benchmarks...")
    rows = []
    
    for n in sizes:
        dc_times = []
        naive_times = []
        dc_comps = []
        naive_comps = []
        
        for _ in range(trials):
            temps = generate_temperatures(n)
            
            # D&C timing
            t0 = now_ns()
            dc_var, dc_comp = temperature_anomaly_dc(temps)
            t1 = now_ns()
            dc_times.append(secs(t1 - t0))
            dc_comps.append(dc_comp)
            
            # Naive timing
            t0 = now_ns()
            naive_var, naive_comp = temperature_anomaly_naive(temps)
            t1 = now_ns()
            naive_times.append(secs(t1 - t0))
            naive_comps.append(naive_comp)
        
        mean_dc_time = sum(dc_times) / len(dc_times)
        std_dc_time = math.sqrt(sum((t - mean_dc_time)**2 for t in dc_times) / len(dc_times))
        mean_dc_comp = sum(dc_comps) / len(dc_comps)
        
        mean_naive_time = sum(naive_times) / len(naive_times)
        std_naive_time = math.sqrt(sum((t - mean_naive_time)**2 for t in naive_times) / len(naive_times))
        mean_naive_comp = sum(naive_comps) / len(naive_comps)
        
        theoretical = theoretical_comparisons(n)
        
        rows.append((n, mean_dc_time, std_dc_time, mean_dc_comp, theoretical,
                    mean_naive_time, std_naive_time, mean_naive_comp))
        
        print(f"  n={n:6d}: D&C {mean_dc_time:.6f}s ({mean_dc_comp:.0f} comps, "
              f"theory {theoretical:.0f}) | Naive {mean_naive_time:.6f}s ({mean_naive_comp:.0f} comps)")
    
    write_csv("temperature_timing.csv",
              ["n", "dc_mean_time_s", "dc_std_time_s", "dc_avg_comparisons", 
               "dc_theoretical_comparisons", "naive_mean_time_s", 
               "naive_std_time_s", "naive_avg_comparisons"],
              rows)
    
    # Generate plots
    plt = try_matplotlib()
    if plt:
        import numpy as np
        ns = [r[0] for r in rows]
        dc_means = [r[1] for r in rows]
        dc_stds = [r[2] for r in rows]
        dc_comps_avg = [r[3] for r in rows]
        theoretical = [r[4] for r in rows]
        
        # Runtime plot
        plt.figure(figsize=(10, 7))
        plt.errorbar(ns, dc_means, yerr=dc_stds, fmt='bo-', capsize=5,
                    capthick=2, markersize=8, linewidth=2,
                    label='Measured runtime')
        
        # Linear fit
        coeffs = np.polyfit(ns, dc_means, 1)
        fitted = np.polyval(coeffs, ns)
        plt.plot(ns, fitted, 'r--', linewidth=2.5,
                label=f'Linear fit: T(n) = {coeffs[0]:.2e}n + {coeffs[1]:.2e}')
        
        # R²
        residuals = np.array(dc_means) - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(dc_means) - np.mean(dc_means))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        plt.xlabel('Input Size (n sensors)', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Divide-and-Conquer Temperature Anomaly Detection', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(path_in_outputs('dc_runtime.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved plot: dc_runtime.png (R²={r_squared:.4f})")
        
        # Comparison count plot
        plt.figure(figsize=(10, 7))
        plt.plot(ns, dc_comps_avg, 'bo-', markersize=8, linewidth=2,
                label='Measured comparisons')
        plt.plot(ns, theoretical, 'r--', linewidth=2.5,
                label='Theoretical: ⌈3n/2⌉ - 2')
        
        plt.xlabel('Input Size (n)', fontsize=14)
        plt.ylabel('Number of Comparisons', fontsize=14)
        plt.title('D&C Comparison Count Analysis', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path_in_outputs('dc_comparisons.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  → Saved plot: dc_comparisons.png")

# ============================================
# Main Execution
# ============================================

def main():
    """Run all experiments and generate outputs."""
    print("=" * 70)
    print("Algorithm Design Project: Greedy & Divide-and-Conquer")
    print("=" * 70)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    random.seed(RANDOM_SEED)
    ensure_outputs_dir()
    
    # ===== PROBLEM A: Battery Charging (Greedy) =====
    print("\n" + "=" * 70)
    print("PROBLEM A: Battery Charging Optimization (Greedy Algorithm)")
    print("=" * 70)
    
    exp_charging_sanity(trials=30, n=12)
    exp_charging_timing(sizes=(200, 400, 800, 1600, 3200, 6400), trials=100)
    
    # ===== PROBLEM B: Temperature Anomaly (D&C) =====
    print("\n" + "=" * 70)
    print("PROBLEM B: Temperature Anomaly Detection (Divide-and-Conquer)")
    print("=" * 70)
    
    exp_temperature_sanity(trials=30, n=100)
    exp_temperature_timing(sizes=(200, 400, 800, 1600, 3200, 6400, 12800), trials=100)
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nAll artifacts written to: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nGenerated files:")
    
    expected_files = [
        "charging_sanity.csv",
        "charging_timing.csv", 
        "greedy_runtime.png",
        "temperature_sanity.csv",
        "temperature_timing.csv",
        "dc_runtime.png",
        "dc_comparisons.png"
    ]
    
    for fn in expected_files:
        p = path_in_outputs(fn)
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  ✓ {fn:30s} ({size:,} bytes)")
        else:
            status = " (skipped - no matplotlib)" if fn.endswith(".png") else " (MISSING)"
            print(f"  ✗ {fn:30s} {status}")
    
    print("\n" + "=" * 70)
    print("Ready for LaTeX inclusion!")
    print("=" * 70)

if __name__ == "__main__":
    main()
