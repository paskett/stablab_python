from __future__ import print_function
import math
import unittest
import numpy as np
np.set_printoptions(precision=15)
from scipy.io import loadmat, savemat,whosmat
port_path = '/Users/joshualytle/bin/projects/pystablab/core/pybvp6c/'

import matplotlib.pyplot as plt
from bvp6c import bvp6c
from bvpinit import bvpinit
from structure_variable import *
from deval import deval
# Import Linear Test Cases
from problems import prob1, prob4, prob7
# Import Nonlinear Test Cases
from problems import prob20, prob27, prob24, prob31, test_ndtools_prob31

import contextlib
import sys
import cStringIO

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout


class TestLinearSystems(unittest.TestCase):
	
	def test_prob1(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob1()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )
		
	
	def test_prob4(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob4()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )
	
	def test_prob7(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob7()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-12 )
		
	def test_prob20(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob20()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )

	def test_prob27(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob27()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )

	def test_prob24(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob24()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-13 and 
		grid_eval < 1e-14 and sol_eval < 1e-13 )
		
	def test_prob31(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = prob31()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )
	
	def test_prob31_numdifftools(self):
		with nostdout():
			(grid_diff, sol_diff, grid_eval, sol_eval) = test_ndtools_prob31()
		
		self.assertTrue(grid_diff < 1e-14 and sol_diff < 1e-14 and 
		grid_eval < 1e-14 and sol_eval < 1e-14 )
	
	
	
	
if __name__ == "__main__":
	unittest.main()