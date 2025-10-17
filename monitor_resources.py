#!/usr/bin/env python3
"""
Resource monitoring script for GPU and CPU usage.
Monitors for a specified duration and reports min, max, and average values.
"""

import subprocess
import time
import re
import psutil
from datetime import datetime

def get_gpu_stats():
    """Get GPU utilization and memory usage from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse: "utilization, memory_used, memory_total"
        parts = result.stdout.strip().split(',')
        gpu_util = float(parts[0].strip())
        gpu_mem_used = float(parts[1].strip())
        gpu_mem_total = float(parts[2].strip())
        return gpu_util, gpu_mem_used, gpu_mem_total
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None, None, None

def get_cpu_stats():
    """Get CPU utilization and memory usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    mem_percent = mem.percent
    return cpu_percent, mem_used_gb, mem_total_gb, mem_percent

def monitor(duration_seconds=120, interval_seconds=1):
    """Monitor resources for specified duration."""
    print(f"Starting resource monitoring for {duration_seconds} seconds...")
    print(f"Sampling every {interval_seconds} second(s)")
    print("-" * 80)

    gpu_utils = []
    gpu_mems = []
    cpu_utils = []
    cpu_mems = []

    start_time = time.time()
    sample_count = 0

    while (time.time() - start_time) < duration_seconds:
        # Get stats
        gpu_util, gpu_mem_used, gpu_mem_total = get_gpu_stats()
        cpu_util, cpu_mem_used, cpu_mem_total, cpu_mem_percent = get_cpu_stats()

        if gpu_util is not None:
            gpu_utils.append(gpu_util)
            gpu_mems.append(gpu_mem_used)

        cpu_utils.append(cpu_util)
        cpu_mems.append(cpu_mem_used)

        sample_count += 1
        elapsed = int(time.time() - start_time)

        # Print current stats every 5 seconds
        if sample_count % 5 == 0:
            print(f"[{elapsed:3d}s] GPU: {gpu_util:5.1f}% ({gpu_mem_used:7.0f}MB) | "
                  f"CPU: {cpu_util:5.1f}% | RAM: {cpu_mem_used:.1f}GB ({cpu_mem_percent:.1f}%)")

        time.sleep(interval_seconds)

    # Calculate statistics
    print("\n" + "=" * 80)
    print("MONITORING SUMMARY")
    print("=" * 80)

    if gpu_utils:
        print(f"\nGPU Utilization:")
        print(f"  Min:     {min(gpu_utils):6.1f}%")
        print(f"  Max:     {max(gpu_utils):6.1f}%")
        print(f"  Average: {sum(gpu_utils)/len(gpu_utils):6.1f}%")

        print(f"\nGPU Memory:")
        print(f"  Min:     {min(gpu_mems):7.0f} MB ({min(gpu_mems)/gpu_mem_total*100:.1f}%)")
        print(f"  Max:     {max(gpu_mems):7.0f} MB ({max(gpu_mems)/gpu_mem_total*100:.1f}%)")
        print(f"  Average: {sum(gpu_mems)/len(gpu_mems):7.0f} MB ({sum(gpu_mems)/len(gpu_mems)/gpu_mem_total*100:.1f}%)")
        print(f"  Total:   {gpu_mem_total:7.0f} MB")

    if cpu_utils:
        print(f"\nCPU Utilization:")
        print(f"  Min:     {min(cpu_utils):6.1f}%")
        print(f"  Max:     {max(cpu_utils):6.1f}%")
        print(f"  Average: {sum(cpu_utils)/len(cpu_utils):6.1f}%")

        print(f"\nSystem RAM:")
        print(f"  Min:     {min(cpu_mems):6.1f} GB")
        print(f"  Max:     {max(cpu_mems):6.1f} GB")
        print(f"  Average: {sum(cpu_mems)/len(cpu_mems):6.1f} GB")
        print(f"  Total:   {cpu_mem_total:6.1f} GB")

    print(f"\nTotal samples: {sample_count}")
    print(f"Duration: {duration_seconds} seconds")
    print("=" * 80)

if __name__ == "__main__":
    monitor(duration_seconds=120, interval_seconds=1)
