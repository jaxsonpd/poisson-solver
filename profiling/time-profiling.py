## 
# @file time-profiling.py
# @author Jack Duignan (Jdu80@uclive.ac.nz)
# @date 2024-10-01
# @brief Run the poisson iteration for various cube sizes

import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime

cube_sizes = [101, 201, 301, 401, 501, 601, 701, 801, 901]

def setup_cmd_args() ->argparse.ArgumentParser:
    """
    Setup the command line arguments that can be provided to the program
    
    ### Params:

    ### Returns:
    out : argparse.ArgumentParser
        The argument parser configured for the application
    """
    parser = argparse.ArgumentParser(description="Profile the poisson calculation program with changing nodes")

    parser.add_argument("-n", "--nodes", nargs="?", action="store", 
                        default=11,
                        help="""max number of nodes to calculate""")
    
    parser.add_argument("-i", "--iterations", nargs="?", action="store",
                        default=300,
                        help="""number of iterations to perform""")
    
    parser.add_argument("-t", "--threads", nargs="?", action="store",
                        default=1,
                        help="""number of threads to use""")
    
    parser.add_argument("-f", "--filename", nargs="?", action="store",
                        default="profile",
                        help="name to append at the beginning of test files")
    
    parser.add_argument("-o", "--object_name", nargs="?", action="store",
                        default="poisson",
                        help="object file to run")
    
    return parser


def execute_poisson(nodes: int, iterations: int, threads: int, object_name: str) -> float:
    """
    Execute one poisson calculation

    ### Params:
    nodes
        Number of nodes to simulate
    iterations
        Number of iterations to complete
    threads
        Number of threads to use
    object_name
        The object file to run

    ### Returns:
    float
        The number of second to perform the iteration
    """
    start_time = time.time()

    os.system(f"../{object_name} -n {nodes} -i {iterations} -t {threads} > /dev/null")

    return time.time() - start_time

def main() -> None:
    parser = setup_cmd_args()
    args = parser.parse_args()

    threads = int(args.threads)
    max_nodes = int(args.nodes)
    iterations = int(args.iterations)
    filename = str(args.filename)
    object_name = str(args.object_name)

    print(f"Starting profiling with up to {max_nodes} nodes, {iterations} iterations, and {threads} threads")

    times = []
    cuda_times = []
    try:
        selected_cubes = cube_sizes[:cube_sizes.index(max_nodes)+1]
    except:
        selected_cubes = cube_sizes
        
    for nodes in selected_cubes:
        times.append(execute_poisson(nodes, iterations, threads, object_name))
        print(f"Node {nodes} executed, time: {times[-1]}")

    with open(f"{filename}_mn{max_nodes}_i{iterations}_t{threads}.csv", "w") as f:
        f.write(f"Time Profile - {datetime.datetime.now()}\n")
        f.write("Cube Size, Time (s)\n")

        for i in range(0, len(selected_cubes)):
            f.write(f"{selected_cubes[i]}, {times[i]}\n")

    plt.plot(selected_cubes, np.array(times), label='CPU')
    plt.title(f"{filename} time poisson n: {max_nodes}, i: {iterations}, t: {threads}")
    plt.xlabel(f"Cube Size")
    plt.ylabel(f"seconds")
    plt.legend()
    # if (max_threads < 5):
    plt.xticks(selected_cubes)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()