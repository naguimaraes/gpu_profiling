#!/bin/bash
# This script runs a given application and monitors the power consumption and GPU memory allocation. Only works with NVIDIA GPUs.

# Edit the following variables to match your desired setup
OUTPUT_PATH="output/"
OUTPUT_FILE_NAME="_profile"
PLOT_PATH="plot/"
LOG_PATH="csv/"
LOG_FILE_NAME="log"
DEFAULT_NUM_RUNS=1
DEFAULT_IDLE_POWER=5.8

PLOT_SCRIPT_COMMAND="python3 " # Make sure to leave a space at the end if you are using a python script
PLOT_SCRIPT="plot.py"
FILE_TO_RUN_COMMAND="./"       # Make sure to NOT leave a space at the end if you are running an executable file
DELAY_MS=10
DELAY_S=$(echo "scale=3; $DELAY_MS / 1000" | bc) 

# Dealing with the input arguments
# Check if the required arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <FILE_TO_RUN> [NUM_RUNS] [-ci | --calculate-idle]. Use -h or --help for more information."
    exit 1
fi

# Check if the user needs help
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "This script runs a given application and monitors the power consumption and GPU memory allocation."
    echo "Usage: $0 <FILE_TO_RUN> [NUM_RUNS] [-ci | --calculate-idle]"
    echo ""
    echo "Arguments:"
    echo "  FILE_TO_RUN: The file to run. This file should be an executable file."
    echo "  NUM_RUNS (optional): The number of runs. Default is 1. More runs provide more accurate results."
    echo "  -ci, --calculate-idle (optional): Before profiling the given application, runs a quick script to calculate the idle power of the GPU."
    exit 0
fi

FILE_TO_RUN=$1  # Get the file to run from the first command-line argument

# Update the files and paths based on the FILE_TO_RUN
OUTPUT_FILE="${OUTPUT_PATH}${FILE_TO_RUN%.*}${OUTPUT_FILE_NAME}.txt"
PLOT_PATH="${PLOT_PATH}${FILE_TO_RUN%.*}"
LOG_PATH="${LOG_PATH}${FILE_TO_RUN%.*}"

if [ $# -lt 2 ]; then
    NUM_RUNS=$DEFAULT_NUM_RUNS  # Set NUM_RUNS to 1 by default if not provided
else
    if [[ $2 =~ ^[0-9]+$ ]]; then
        NUM_RUNS=$2  # Get NUM_RUNS from the second command-line argument
    else
        NUM_RUNS=$DEFAULT_NUM_RUNS  # Set NUM_RUNS to 1 by default if not provided
    fi
fi

# Check if the user wants to calculate the idle power
if [[ $3 == "-ci" || $3 == "--calculate-idle" || $2 == "-ci" || $2 == "--calculate-idle" ]]; then
    echo "Calculating idle power before profiling ${FILE_TO_RUN}..."

    # Wait for the power to stabilize
    sleep 15

    echo "Warning! Killing all user-created processes running on the GPU..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk -F',' '{if ($2 == "C") print $1}' | xargs -I {} kill -9 {}

    # Measure idle power multiple times and take the average
    IDLE_POWER=0
    NUM_MEASUREMENTS=5000 
    for ((i=1; i<=NUM_MEASUREMENTS; i++))
    do
        # Get the instant power consumption
        POWER_DRAW=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader| head -n 1 | tr -d '[:space:]' | sed 's/W//')
        
        # Accumulate the power measurements
        IDLE_POWER=$(echo "$IDLE_POWER + $POWER_DRAW" | bc)
        
        # Wait one iteration from nvidia-smi before checking the power again
        sleep $DELAY_S # 10ms
    done
    # Calculate the average idle power
    IDLE_POWER=$(echo "scale=2; $IDLE_POWER / $NUM_MEASUREMENTS" | bc)
    echo "Calculated idle power: $IDLE_POWER W"
else
    IDLE_POWER=$DEFAULT_IDLE_POWER 
fi


# Arrays to store power consumption, GPU memory allocation, and execution time for each run
POWER_ARRAY=()
MEMORY_ARRAY=()
TIME_ARRAY=()

# Create and clear the LOG_PATH folder if it already exists
if [ -d "$LOG_PATH" ]; then
    rm -rf "$LOG_PATH"/*
else
    mkdir -p "$LOG_PATH"
fi

# Create and clear the memory/ folder if it already exists
MEMORY_FOLDER="$LOG_PATH/memory"
if [ -d "$MEMORY_FOLDER" ]; then
    rm -rf "$MEMORY_FOLDER"/*
else
    mkdir -p "$MEMORY_FOLDER"
fi

# Create and clear the power/ folder if it already exists
POWER_FOLDER="$LOG_PATH/power"
if [ -d "$POWER_FOLDER" ]; then
    rm -rf "$POWER_FOLDER"/*
else
    mkdir -p "$POWER_FOLDER"
fi

# Create and clear the PLOT_PATH folder if it already exists
if [ -d "$PLOT_PATH" ]; then
    rm -rf "$PLOT_PATH"/*
else
    mkdir -p "$PLOT_PATH"
fi

# Create the OUTPUT_PATH folder if it does not exist
if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
fi

# Variables used in the script, do not change
TOTAL_POWER=0
TOTAL_TIME=0
TOTAL_MEMORY=0
AVERAGE_POWER=0
AVERAGE_TIME=0
AVERAGE_MEMORY=0

# Clear the output file
echo "Run, Average Power (Watts), Memory Allocation (MiB), Time (Milliseconds)" > $OUTPUT_FILE

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Running iteration $i: "
    POWER_LOG_FILE="${LOG_PATH}/power/${LOG_FILE_NAME}$i.csv"
    MEMORY_LOG_FILE="${LOG_PATH}/memory/${LOG_FILE_NAME}$i.csv"

    echo "Warning! Killing all user created processes running on the GPU..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk -F',' '{if ($2 == "C") print $1}' | xargs -I {} kill -9 {}

    # Start monitoring power and GPU memory allocation
    nvidia-smi --query-gpu=power.draw --format=csv,noheader --loop-ms=${DELAY_MS} &> $POWER_LOG_FILE &
    POWER_MONITORING_PID=$!
    nvidia-smi --query-gpu=memory.used --format=csv,noheader --loop-ms=${DELAY_MS} &> $MEMORY_LOG_FILE &
    MEMORY_MONITORING_PID=$!

    # Start the timer and run the application
    START_TIME=$(date +%s%N)
    $FILE_TO_RUN_COMMAND$FILE_TO_RUN
    END_TIME=$(date +%s%N)

    echo "Execution complete. Waiting for the power to stabilize..."

    while true; do
        # Get the instant power consumption and GPU memory allocation
        POWER_DRAW=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader| head -n 1 | tr -d '[:space:]' | sed 's/W//')
        
        # Check if the instant power read is lower than the idle power
        if (( $(echo "($POWER_DRAW - 1) < $IDLE_POWER" | bc -l) )); then
            # Stop monitoring power and GPU memory allocation
            kill $POWER_MONITORING_PID
            kill $MEMORY_MONITORING_PID
            echo "Power stabilized."
            break
        fi

        # Wait one iteration from nvidia-smi before checking the power and GPU memory again
        sleep $DELAY_S # 10ms
    done

    # Calculate average power consumption from the log
    AVG_POWER=$(awk '{s+=$1} END {print s/NR}' $POWER_LOG_FILE)
    AVG_POWER_NUM=$(echo "$AVG_POWER" | sed 's/,/./')
    
    # Calculate average GPU memory allocation from the log
    AVG_MEMORY=$(awk '{s+=$1} END {print s/NR}' $MEMORY_LOG_FILE)
    AVG_MEMORY_NUM=$(echo "$AVG_MEMORY" | sed 's/,/./')
    
    # Calculate execution time in milliseconds
    EXECUTION_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

    # Log power consumption, GPU memory allocation, and execution time
    echo "$i, $AVG_POWER_NUM, $AVG_MEMORY_NUM, $EXECUTION_TIME" >> $OUTPUT_FILE
    TOTAL_POWER=$(echo "$TOTAL_POWER + $AVG_POWER_NUM" | bc)
    TOTAL_TIME=$(echo "$TOTAL_TIME + $EXECUTION_TIME" | bc)
    TOTAL_MEMORY=$(echo "$TOTAL_MEMORY + $AVG_MEMORY_NUM" | bc)

    # Store power consumption, GPU memory allocation, and execution time in arrays
    POWER_ARRAY+=($AVG_POWER_NUM)
    MEMORY_ARRAY+=($AVG_MEMORY_NUM)
    TIME_ARRAY+=($EXECUTION_TIME)

    # Plot the power consumption
    echo "Plotting power consumption for iteration $i..."
    $PLOT_SCRIPT_COMMAND$PLOT_SCRIPT $EXECUTION_TIME $i $POWER_LOG_FILE $PLOT_PATH $DELAY_S


    # Plot the memory allocation
    echo "Plotting memory allocation for iteration $i..."
    $PLOT_SCRIPT_COMMAND$PLOT_SCRIPT $EXECUTION_TIME $i $MEMORY_LOG_FILE $PLOT_PATH $DELAY_S

    echo "Iteration $i complete."
    echo ""

    # Wait for a second before starting the next run
    sleep 1
done

# Calculate overall averages
AVERAGE_POWER=$(echo "scale=2; $TOTAL_POWER / $NUM_RUNS" | bc)
AVERAGE_MEMORY=$(echo "scale=2; $TOTAL_MEMORY / $NUM_RUNS" | bc)
AVERAGE_TIME=$(echo "scale=2; $TOTAL_TIME / $NUM_RUNS" | bc)

# Calculate standard deviation for power consumption
POWER_SUM=0
for power in "${POWER_ARRAY[@]}"; do
    DIFF=$(echo "$power - $AVERAGE_POWER" | bc)
    SQUARE=$(echo "$DIFF^2" | bc)
    POWER_SUM=$(echo "$POWER_SUM + $SQUARE" | bc)
done
POWER_STDDEV=$(echo "scale=2; sqrt($POWER_SUM / $NUM_RUNS)" | bc)

# Calculate standard deviation for GPU memory allocation
MEMORY_SUM=0
for memory in "${MEMORY_ARRAY[@]}"; do
    DIFF=$(echo "$memory - $AVERAGE_MEMORY" | bc)
    SQUARE=$(echo "$DIFF^2" | bc)
    MEMORY_SUM=$(echo "$MEMORY_SUM + $SQUARE" | bc)
done
MEMORY_STDDEV=$(echo "scale=2; sqrt($MEMORY_SUM / $NUM_RUNS)" | bc)

# Calculate standard deviation for execution time
TIME_SUM=0
for time in "${TIME_ARRAY[@]}"; do
    DIFF=$(echo "$time - $AVERAGE_TIME" | bc)
    SQUARE=$(echo "$DIFF^2" | bc)
    TIME_SUM=$(echo "$TIME_SUM + $SQUARE" | bc)
done
TIME_STDDEV=$(echo "scale=2; sqrt($TIME_SUM / $NUM_RUNS)" | bc)

# Append overall averages and standard deviations to the output file
echo "Average Power Consumption Through $NUM_RUNS run(s): ($AVERAGE_POWER ± $POWER_STDDEV) W" >> $OUTPUT_FILE
echo "Average GPU Memory Allocation Through $NUM_RUNS run(s): ($AVERAGE_MEMORY ± $MEMORY_STDDEV) MiB" >> $OUTPUT_FILE
echo "Average Execution Time Through $NUM_RUNS run(s): ($AVERAGE_TIME ± $TIME_STDDEV) ms" >> $OUTPUT_FILE

echo "Power profiling complete. Results saved in $OUTPUT_FILE."
echo "Plots saved in $PLOT_PATH directory and log files saved in $LOG_PATH directory."