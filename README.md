# Power and Memory Profiling of GPU Programs

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Before running the script](#before-running-the-script)
4. [Usage](#usage)
5. [Outputs](#outputs)
6. [Contribuitions](#contribuitions)
7. [Known Issues](#known-issues)

## Introduction

This repository contains a bash script that measures the power consumption and the memory allocation of a given program running on an NVIDIA GPU, which can be useful for energy efficiency analysis or to check if the program is running as expected.

After putting to run the program you want to profile, the script will start measuring both the power consumption and the memory allocation of it in fixed intervals. Even if the program has finished its execution, the script will keep measuring the GPU until the power consumption and memory allocation reach their idle state.

The read data is after saved in a csv file and can be used to analyze the results. The script also provides a summary of the runs, with the average power consumption and average memory allocation of the program and a plot of the csv data.

The script is built on top of the [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface), which is the API used to monitor and manage the usage of the GPU.

This work was developed for the Embedded Systems class (INF01059) at the [Federal University of Rio Grande do Sul (UFRGS)](www.ufrgs.br/) with the help of [Eduardo](https://github.com/BoslondeHiggs) and [Pedro](https://github.com/PedroMiola).

## Installation

1. Clone the repository

    ```sh
    git clone https://github.com/naguimaraes/gpu_profiling.git
    ```

2. Enter the repository directory

    ```sh
    cd gpu_profiling
    ```

3. Install the dependencies

    ```sh
    chmod +x dependencies.sh
    ./dependencies.sh
    ```

## Before running the script

1. Make sure your GPU is supported by the nvidia-smi tool. You can check it [here](https://developer.nvidia.com/nvidia-system-management-interface).

2. Make sure your program **runs** and that it **will run on the GPU**, by using CUDA, OpenCL or any other GPU programming API. Otherwise, the script can be stuck in an **infinite loop**.

3. Make sure your program is ready to run via command line:
    - If it is a compiled executable, you can run it directly with no problems.
    - If it is, for example, a python program, you have to add `#!/usr/bin/env python3` at the beginning of the script and give it execution permission with `chmod +x program.py`.

4. Make sure your program runs long enough to be measured. The script will keep measuring the GPU until it reaches the idle state, so if your program runs for a very short time, the script will keep measuring the GPU for a long time.

## Usage

First, make the script executable with the following command:

```sh
chmod +x gpu_profiling.sh
```

Then, you can check all the arguments and their usage by running:

```sh
./gpu_profiling.sh --help
```

The simplest way to use the script is to just run it with the program you want to profile as the only argument. For example, if you want to profile a python program called `program.py`, you can run the following command:

```sh
./gpu_profiling.sh program.py
```

But the script support other arguments as well, such as profiling it multiple times or calculate the idle power consumption of the GPU. For example, if you want to profile the program 5 times, you can run:

```sh
./gpu_profiling.sh program.py 5
```

Running it more times will give you a better average of the power consumption and memory allocation of the program. The script will run the program as many times as you want, waiting for the GPU to reach the idle state between each run.

If you want to calculate the idle power consumption of the GPU, you can run:

```sh
./gpu_profiling.sh program.py -ci
```

or

```sh
./gpu_profiling.sh program.py --calculate-idle
```

You can also insert (hard-coded) the idle power consumption of the GPU in the script, but it is not recommended, as the idle power consumption can change depending on the GPU model and the system configuration. So, its better to calculate it using the script each time you want to profile a program.

## Outputs

The first lines of the `gpu_profiling.sh` can be modified to change where the outputs and results of the script will be saved. By default, it will save:

- the data read from the GPU into a csv inside the `csv/[program_filename]/` directory;
- a graph plot of the same data inside the `plot/[program_filename]/` directory and
- a summary of the runs into a text file inside the `output/[program_filename]_results.txt` file.

## Contribuitions

We added, in the `contrib/MNIST` directory, a test case that we used to validate the script. It is a comparison between 3 different deep learning frameworks (TensorFlow, PyTorch and Chainer) running the same neural network model (MNIST) with the same hyperparameters, to check the differences in power consumption and memory allocation between them. The python codes used are inside the 'root' directory and the results are inside the `output/`, `csv/` and `plot/` directories.

We are open to suggestions on the script and also to add new test cases, such as ours, to the `contrib/` folder. If you want to contribute with this project, you can open an issue or a pull request. For any other questions, feel free to contact me at <naguimaraes@inf.ufrgs.br>.

We hope this script can be useful for you and your projects. Enjoy!

## Known Issues

- If your program was written in Windows and you are running it in a Linux environment, you may have to change the line endings of the script to LF. You can do this with the following command:

    ```sh
    sudo apt install dos2unix
    dos2unix dependencies.sh
    dos2unix gpu_profiling.sh
    dos2unix your_program
    ```
