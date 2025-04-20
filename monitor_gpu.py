"""
Script to monitor GPU usage during processing.
"""

import time
import argparse
import subprocess
import threading
import csv
import os
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor GPU usage during processing")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds")
    parser.add_argument("--output", type=str, default="gpu_usage.csv",
                        help="Output CSV file")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration to monitor in seconds (None for indefinite)")
    return parser.parse_args()

def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n")
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

def monitor_gpu(interval, output_file, duration=None):
    """Monitor GPU usage and write to CSV file."""
    start_time = time.time()
    
    # Create CSV file and write header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Elapsed",
            "GPU_Index",
            "GPU_Name",
            "Temperature",
            "GPU_Utilization",
            "Memory_Utilization",
            "Memory_Used",
            "Memory_Total",
            "Power_Draw"
        ])
    
    print(f"Monitoring GPU usage every {interval} seconds. Press Ctrl+C to stop.")
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if duration limit reached
            if duration is not None and elapsed > duration:
                print(f"Monitoring completed after {elapsed:.2f} seconds")
                break
            
            # Get GPU information
            gpu_info = get_gpu_info()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Write to CSV file
            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                
                if not gpu_info:
                    writer.writerow([timestamp, f"{elapsed:.2f}", "N/A", "No GPU found", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                    print(f"[{timestamp}] No GPU found")
                else:
                    for gpu in gpu_info:
                        gpu_data = gpu.split(", ")
                        writer.writerow([timestamp, f"{elapsed:.2f}"] + gpu_data)
                        
                        # Print current usage
                        if len(gpu_data) >= 8:
                            gpu_index, gpu_name, temp, gpu_util, mem_util, mem_used, mem_total, power = gpu_data
                            print(f"[{timestamp}] GPU {gpu_index}: {gpu_util}% util, {mem_used}/{mem_total} MB, {temp}°C, {power}W")
                        else:
                            print(f"[{timestamp}] GPU data: {gpu_data}")
            
            # Wait for next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped after {time.time() - start_time:.2f} seconds")
        print(f"Results saved to {output_file}")

def run_command_with_monitoring(command, interval=1.0, output_file="gpu_usage.csv"):
    """Run a command while monitoring GPU usage."""
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(interval, output_file),
        daemon=True
    )
    monitor_thread.start()
    
    # Run the command
    try:
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
        print("Command completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nCommand interrupted")
    
    # Stop monitoring
    print("Stopping GPU monitoring...")
    time.sleep(interval * 2)  # Give time for final measurements

if __name__ == "__main__":
    args = parse_args()
    monitor_gpu(args.interval, args.output, args.duration)
