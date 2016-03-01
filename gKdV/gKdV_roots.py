import sys
import pickle
import os
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

def movie_maker(record, moviename,title):
	File = open(record,'rb')
	REC = pickle.load(File)
	File.close()
	
	os.makedirs('roots_temp')
	
	for j in range(len(REC[:,0])):
		plt.plot(REC[0:j+1,0], REC[0:j+1,1],'-k',linewidth=2.0)
		plt.plot(REC[j,0], REC[j,1],'.k')	
		plt.title(title)
		plt.axis([5,10,0,9])
		plt.xlabel(r'$p$',fontsize=18)
		plt.ylabel(r'$\lambda$',fontsize=18)
		if j >9:
			filestring = ('%i'%j)
		else:
			filestring = '0'+('%i'%j)
		print j
		plt.savefig('roots_temp/'+filestring+'.png')
		plt.clf()
	
	current_path = os.getcwd()
	os.chdir(current_path+'/'+'roots_temp')
	
	call("ffmpeg -r 9 -i %02d.png -r 25 -q:v 1 "+moviename,shell=True)
	list = os.listdir(os.getcwd())
	for item in list:
		if '.png' in item:
			os.remove(item)
	
	os.rename(moviename,current_path+'/'+moviename)
	os.chdir(current_path)
	os.rmdir(current_path+'/'+'roots_temp')
	return


def plot_gKdV_roots():
	File = open('gKdV_Data/finished_rec.pkl','rb')
	REC = pickle.load(File)
	File.close()
	
	plt.plot(REC[:,0], REC[:,1],'-k',linewidth=2.0)
	plt.plot(REC[:,0], REC[:,1],'.k')	
	plt.title('Eigenvalues for the gKdV equation,\n computed by bvp\_solver.')
	plt.axis([5,10,0,9])
	plt.xlabel(r'$p$',fontsize=16)
	plt.ylabel(r'$\lambda$',fontsize=16)
	plt.savefig('gKdV_Data/gKdV_roots.pdf')
	plt.show()
	plt.clf()
	return 



record = 'gKdV_Data/finished_rec.pkl'
moviename = 'gKdV_roots.mpg'
title = 'Eigenvalues for the gKdV equation.\n Computed by bvp\_solver.'
movie_maker(record, moviename,title)

# plot_gKdV_roots()