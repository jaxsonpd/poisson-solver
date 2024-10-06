## 
# @file plot-results.py
# @author Jack Duignan (Jdu80@uclive.ac.nz)
# @date 2024-10-02
# @brief Plot the results of several trials

import matplotlib.pyplot as plt
import numpy as np

from typing import List

def plot_results(filenames: List[str], labels: List[str], plot_type: str = "time") -> plt.Figure:
    """
    Plot several results on the same figure

    ### Params:
    filenames: List[str]
     A list of csv filenames
    labels: List[str]
     Labels for each data set
    plot_type: str
     Type of plot (time or thread) case sensitive
    """

    figure = plt.figure()

    for file, label in zip(filenames, labels):
        data = np.loadtxt(file, delimiter=",", skiprows=2)
        # You can customize plotting based on the type if needed
        if (filenames.index(file) == 2):
            plt.plot(data[:, 0], data[:, 1], label=label, linestyle="--")
        else:
            plt.plot(data[:, 0], data[:, 1], label=label)

        

    plt.grid(True)
    if (plot_type == "thread"):
        plt.xlabel("Number of Threads")
    else:
        plt.xlabel("Cube Size")
    
    plt.ylabel("Time (s)")
    plt.xlim(1, None)
    plt.ylim(1, None)
    plt.legend()  # Add a legend for the labels
    plt.tight_layout()
    
    return figure


if __name__ == "__main__":
    figure = plot_results(["./Compilier-Opt/JPC_03_mn901_i300_t20.csv", 
                           "./CUDA-cmp/gpu_mn601_i300_t1.csv"],
                           ["Intel 12400F CPU -O3", "NVIDIA 3070 Ti Mobile GPU"], plot_type="time")

    figure.savefig("JPC_CPU_Razer_GPU.png")
    plt.show()