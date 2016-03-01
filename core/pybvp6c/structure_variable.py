from functools import wraps
import numpy as np
from numpy.linalg import norm
from types import IntType, LongType, FloatType

eps = np.finfo(np.float).eps
# print "eps = ", eps

def isnumeric(n):
	return isinstance(n,(IntType,LongType,FloatType))

class struct():
	pass

def print_this(func):
	@wraps(func)
	def wrapper(*args,**kwargs):
		name = func.__name__
		str_02 = '-----------------------------------------------'
		enter_name = 'Entering '+name
		fill = (( len(str_02)-len('Entering '+name) )//2)*'-'
		print(str_02)
		print(fill+enter_name+fill)
		# print(str_02)
		print(2*'\n')
		
		result = func(*args,**kwargs)
		
		leave_name = 'Leaving '+name
		fill = (( len(str_02)-len('Leaving '+name) )//2)*'-'
		# print(str_02)
		print(fill+leave_name+fill)
		print(str_02)
		print(2*'\n')
		return result
	return wrapper


# @contextmanager
# def nostdout():
# 	save_stdout = sys.stdout
# 	sys.stdout = StringIO()
# 	yield
# 	sys.stdout = save_stdout
#
# saved_stdout = sys.stdout
