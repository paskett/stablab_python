import numpy as np
from scipy.interpolate import interp1d



def init_eigf_guess(z,x,f_valf_val):
	y = np.empty(f_val.shape[0])
	for j in range(0,f_val.shape[0]):
		y[j] = interp1d(x, np.real(f_val[j,:]), kind='linear')(z)
	return y
