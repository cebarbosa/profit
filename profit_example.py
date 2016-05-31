# -*- coding: utf-8 -*-
"""

Created on 31/05/16

@author: Carlos Eduardo Barbosa

Simple application of the profit routine.

"""
import matplotlib.pyplot as plt

import profit as p

if __name__ == "__main__":
    # Seting the input model and the output file
    infile = "input_5251.txt"
    outfile = "output_5251.txt"
    # Initialize the models
    M = p.Profit(infile, outfile)
    # Run fitting model
    M.fit()
    # Plotting routine
    M.plot()
    # plt.show(block=True) # To visualize the model
    # plt.savefig("example.png") # To save the figure
    # Calculate errors
    # Errors in the parameters are not correct in the program. So, instead
    # once you finish with a model (not minding with the errors), you can
    # reliable estimate the errors using the routine bellow.
    p.bootstrap("output_5251.txt", logfile="bootlog.txt", nsim=30)
    p.update_errors("output_5251.txt", "bootlog.txt")