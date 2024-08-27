import argparse
import pandas as pd
import os

import matplotlib.pyplot as plt

def plot_watts_time_from_file(file_path, run_number, real_runtime, plot_folder, time_interval):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the Watts values and create a list of times (assuming each measurement is 100ms apart)
    watts = [float(line.split()[0]) for line in lines]
    times = [i * time_interval for i in range(len(watts))]  # time in seconds

    # Create a DataFrame
    data = pd.DataFrame({'Time': times, 'Watts': watts})
    
    # Plot Watts over Time
    plt.figure(figsize=(19.2, 10.8))  # Set the figure size to Full HD (1920x1080)
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.plot(data['Time'], data['Watts'], lw=2, linestyle='-', color='#2f4f7f')
    plt.title('GPU Power Consumption Over Time (Run {})'.format(run_number), fontsize=24, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Power (W)', fontsize=22)
    plt.grid(True, linestyle='-', alpha=0.7)

    # Add vertical lines with captions
    plt.axvline(x=times[0], color='#2e865f', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(times[0], max(watts)*0.9, 'Code started executing.', color='#2e865f', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)
    
    plt.axvline(x=real_runtime, color='#2e865f', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(real_runtime, max(watts)*0.9, 'Code ended its execution.', color='#2e865f', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)

    plt.axvline(x=times[-1], color='#e74c3c', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(times[-1], max(watts)*0.9, 'Power consumption stabilized.', color='#e74c3c', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)

    # Create the plots directory if it doesn't exist
    plot_folder = f'{plot_folder}/power'
    os.makedirs(f'{plot_folder}', exist_ok=True)

    # Save the plot as a PNG image in the plots directory
    plot_file_path = os.path.join(f'{plot_folder}', 'plot{}.png'.format(run_number))
    plt.savefig(plot_file_path, dpi=300)  # Set the dpi to 100 for Full HD resolution
    
def plot_memory_time_from_file(file_path, run_number, real_runtime, plot_folder, time_interval):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the Memory values and create a list of times (assuming each measurement is 100ms apart)
    memory = [float(line.split()[0]) for line in lines]
    times = [i * time_interval for i in range(len(memory))]  # time in seconds

    # Create a DataFrame
    data = pd.DataFrame({'Time': times, 'Watts': memory})
    
    # Plot Watts over Time
    plt.figure(figsize=(19.2, 10.8))  # Set the figure size to Full HD (1920x1080)
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.plot(data['Time'], data['Watts'], lw=2, linestyle='-', color='#2f4f7f')
    plt.title('GPU Memory Allocation Over Time (Run {})'.format(run_number), fontsize=24, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Memory (MiB)', fontsize=22)
    plt.grid(True, linestyle='-', alpha=0.7)

    # Add vertical lines with captions
    plt.axvline(x=times[0], color='#2e865f', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(times[0], max(memory)*0.9, 'Code started executing.', color='#2e865f', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)
    
    plt.axvline(x=real_runtime, color='#2e865f', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(real_runtime, max(memory)*0.9, 'Code ended its execution.', color='#2e865f', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)

    plt.axvline(x=times[-1], color='#e74c3c', lw=2, linestyle='--', ymin=0, ymax=1)
    plt.text(times[-1], max(memory)*0.9, 'Power consumption stabilized.', color='#e74c3c', verticalalignment='center_baseline', ha='left', rotation=270, fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)

    # Create the plots directory if it doesn't exist
    plot_folder = f'{plot_folder}/memory'
    os.makedirs(f'{plot_folder}', exist_ok=True)

    # Save the plot as a PNG image in the plots directory
    plot_file_path = os.path.join(f'{plot_folder}', 'plot{}.png'.format(run_number))
    plt.savefig(plot_file_path, dpi=300)  # Set the dpi to 100 for Full HD resolution

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Plot GPU power consumption over time.')
    parser.add_argument('real_runtime', type=int, help='Time in seconds the code took to run')
    parser.add_argument('run_number', type=int, default=1 ,help='Number of the run to plot')
    parser.add_argument('file_path', type=str, default='csv', help='Path to the file containing power consumption data')
    parser.add_argument('plot_folder', type=str, default='', help='Path to save the plot file')
    parser.add_argument('time_interval', type=float, default=0.01, help='Time interval, in seconds, between measurements')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    args.real_runtime = args.real_runtime/1000 # Convert ms to s
    
    # Call the plot function with the provided file path and run number
    if 'memory' in args.file_path:
        plot_memory_time_from_file(args.file_path, args.run_number, args.real_runtime, args.plot_folder, args.time_interval)
    else:
        plot_watts_time_from_file(args.file_path, args.run_number, args.real_runtime, args.plot_folder, args.time_interval)
