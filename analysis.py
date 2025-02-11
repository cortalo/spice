#!/usr/bin/env python3

from parser import *
from stamp import *
import numpy

def init_matrix(elements, matrix, rhs, models):
    for element in elements.values():
        stamp(matrix, rhs, element, models=models)

def dc(elements, models):
    all_nodes = get_all_nodes(elements)
    group2_indices = get_group2_indices(elements, len(all_nodes))
    nonlinear_indices = get_nonlinear_indices(elements)
    matrix_dim = len(all_nodes)-1+len(group2_indices)
    matrix = numpy.zeros((matrix_dim,matrix_dim))
    rhs = numpy.zeros((matrix_dim,1))
    init_matrix(elements, matrix, rhs, models=models)
    if nonlinear_indices == []:
        print(numpy.linalg.solve(matrix, rhs))
    else:
        prev_sol = [0 for _ in range(matrix_dim)]
        cur_sol = [float(item[0]) for item in numpy.linalg.solve(matrix, rhs)]
        for i in range(10):
            for j in nonlinear_indices:
                stamp(matrix, rhs, elements.values()[j], positive=False, sol=prev_sol, models=models)
                stamp(matrix, rhs, elements.values()[j], positive=True, sol=cur_sol, models=models)
            prev_sol = cur_sol
            cur_sol = [float(item[0]) for item in numpy.linalg.solve(matrix, rhs)]
        print(cur_sol)
