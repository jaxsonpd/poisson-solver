## 
# @file profiling.py
# @author Jack Duignan (Jdu80@uclive.ac.nz)
# @date 2024-09-26
# @brief Profile the solution testing for various thread counts etc.

import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np

def setup_cmd_args() ->argparse.ArgumentParser:
    """
    Setup the command line arguments that can be provided to the program
    
    ### Params:

    ### Returns:
    out : argparse.ArgumentParser
        The argument parser configured for the application
    """
    parser = argparse.ArgumentParser(description="Profile the poisson calculation program")

    parser.add_argument("-n", "--nodes", nargs="?", action="store", 
                        default=11,
                        help="""number of nodes to calculate""")
    
    parser.add_argument("-i", "--iterations", nargs="?", action="store",
                        default=300,
                        help="""number of iterations to perform""")
    
    parser.add_argument("-t", "--threads", nargs="?", action="store",
                        default=1,
                        help="""max number of threads to try""")
    
    parser.add_argument("-f", "--filename", nargs="?", action="store",
                        default="profile",
                        help="name to append at the beginning of test files")
    
    return parser


def execute_poisson(nodes: int, iterations: int, threads: int) -> float:
    """
    Execute one poisson calculation

    ### Params:
    nodes
        Number of nodes to simulate
    iterations
        Number of iterations to complete
    threads
        Number of threads to use

    ### Returns:
    float
        The number of second to perform the iteration
    """
    start_time = time.time()

    os.system(f"./poisson -n {nodes} -i {iterations} -t {threads} > /dev/null")

    return time.time() - start_time


def main() -> None:
    parser = setup_cmd_args()
    args = parser.parse_args()

    max_threads = int(args.threads)
    nodes = int(args.nodes)
    iterations = int(args.iterations)
    filename = str(args.filename)

    print(f"Starting profiling with {nodes} nodes, {iterations} iterations, and up to {max_threads} threads")

    times = []
    for thread in range(1, max_threads):
        times.append(execute_poisson(nodes, iterations, thread))
        if (thread % (max_threads/10) == 0):
            print(f"Thread {thread} Executed")

    with open(f"{filename}_n{nodes}_i{iterations}_t{max_threads}.csv", "w") as f:
        f.write("Number Threads, Time (s)\n")

        for i in range(1, max_threads):
            f.write(f"{i}, {times[i-1]}\n")

    plt.plot(np.arange(1, max_threads), np.array(times))
    plt.title(f"{filename} poisson n: {nodes}, i: {iterations}, t: {max_threads}")
    plt.xlabel(f"num threads")
    plt.ylabel(f"seconds")
    if (max_threads < 5):
        plt.xticks(np.arange(max_threads))

    plt.show()


if __name__ == "__main__":
    main()