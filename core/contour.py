import os

import matplotlib.pyplot as plt
import numpy as np



def analytic_basis(projection,x,preimage,s,p,A,posneg,eps):
	iterations = preimage.shape[0]
	matrix = A(x,preimage[0],s,p)
	p_old, Q1 = projection(A(x,preimage[0],s,p),posneg,eps)
	
	n, k = Q1.shape
	# print "lambda = ", preimage[0]
	# print "shape of basis matrix = ", Q1.shape
	out = np.zeros((n,k,iterations),dtype='complex')
	projects = np.zeros((p_old.shape[0],p_old.shape[1],iterations),dtype='complex')
	
	if (np.imag(preimage[0]) == 0):
		p_old = np.real(p_old)
		Q1 = np.real(Q1)
	
	out[:,:,0] = Q1
	projects[:,:,0] = p_old
	for j in np.arange(1,iterations):
		proj,temp = projection(A(x,preimage[j],s,p),posneg,eps)
		# if not temp.shape[1] == 4:
		# 	print "lambda = ", preimage[j]
		# 	print "shape of basis matrix = ", temp.shape
		# 	raise SystemError
		#%out(:,:,j) = proj * (eye(n) + proj * p_old - p_old * proj) * out(:,:,j-1)
		out[:,:,j] = (proj.dot(np.eye(n) + 0.5 * p_old.dot(np.eye(n)- proj))).dot(out[:,:,j-1])
		projects[:,:,j] = proj
		p_old = proj
	return out, projects


def winding_number(w):
	'''
	Returns the winding number of a contour w
	
	Input "w" should be a closed contour not passing through zero.
	
	If f is analytic and nonzero at each point of a simple closed positively
	oriented contour C and is meromorphic inside C, then
	winding_number(w)=N0-Np where w=f'(C)./f(C) and N0 and Np are respectively
	the number of zeros and poles of f inside C (multiplicity included).
	
	The change in the argument between any two points of w should be less than
	Pi for winding_number(w) to be accurate.
	
	Computes the winding number of the contour'''
	
	out = 0
	for k in range(len(w)-1):
		if np.imag(w[k+1])==0 and np.real(w[k+1]) < 0: 
			kp=np.pi*np.sign(np.imag(w[k]))
		else:
			kp=np.imag( np.log(np.true_divide( w[k+1],abs(w[k+1]) )) )
			
		if np.imag(w[k]) == 0 and np.real(w[k]) < 0:
			kc = np.pi*np.sign(np.imag(w[k+1]))
		else:
			kc=np.imag( np.log(np.true_divide( w[k],abs(w[k]) )) )
			
		opt1 = kp-kc
		opt2 = -(2*np.pi-abs(opt1))*np.sign(opt1);
		
		if min(abs(opt1),abs(opt2))==abs(opt1):
			out=out+opt1
		else:
			out=out+opt2
		
	return round(np.true_divide(out,(2*np.pi) ))

	
def semicirc2(circpnts,imagpnts,innerpnts,ksteps,r,spread,inner_radius,lambda_steps):
	'''
	semicirc2(circpnts,imagpnts,innerpnts,ksteps,R,spread,inner_radius,lambda_steps):
	return preimage
	Returns a quarter of a semicircle with a small inner circle skirting the
	origin.
	
	Input "circpnts" is the number of points on the circle part, "imagpnts"
	is the number of points on the imaginary axis, "ksteps" is the number of
	kato steps taken between points, "R" is the
	radius of the semicircle, "spread" is a constant that spreads the points
	on the imaginary axis so that they are more dense near the origin, and
	"inner_radius" is how close along the imaginary axis the contour comes to the
	origin. If not specified, "lambda_points" defaults to zero. If specified,
	"lambda_points" is the number of contour points between Kato steps. That
	is, the analytic basis is not computed on the additional lambda_steps
	points unless achieving relative error tolerances of the Evans function ouput
	requires it. If that is the case, the analytic Kato basis is only
	computed on the region where additional Evans function evaluations are
	needed.
	'''
	
	# Specify points
	p1=(circpnts-1)*ksteps+circpnts + (((circpnts-1)*ksteps+circpnts)-1)*lambda_steps
	p2=(imagpnts-1)*ksteps+imagpnts+ksteps+1+(((imagpnts-1)*ksteps+imagpnts+ksteps+1)-1)*lambda_steps
	p3=innerpnts*ksteps+innerpnts+1 + ((innerpnts*ksteps+innerpnts+1)-1)*lambda_steps
	
	# construct and combine contour parts
	theta = np.linspace(0,np.pi/2,p1)
	k = spread**(-1)
	ln=np.power( np.linspace(r**k,inner_radius**k,p2) ,spread)
	theta2=np.linspace(np.pi/2,0,p3)
	preimage=np.concatenate(( r*np.exp(theta*1j),ln[1:]*1j,inner_radius*np.exp(theta2[1:]*1j) ))
	
	return preimage


def Evans_plot(w,labelstring,titlestring, filestring, format,Plot_B=False):
	"""	
	Taken from blog post
	http://blog.olgabotvinnik.com/post/58941062205/prettyplotlib-painlessly-create-beautiful-matplotlib

	# Save a nice dark grey as a variable
	almost_black = '#262626'
	"""
	
	fig, ax = plt.subplots(1)
	xl, xr = np.min(np.real(w)), np.max(np.real(w))
	yl, yr = np.min(np.imag(w)), np.max(np.imag(w))
	ylength, xlength = yr - yl, xr - xl
	ybuffer, xbuffer = ylength/8., xlength/8.
	ax.set_ylim([yl-ybuffer,yr+ybuffer])
	ax.set_xlim([xl-xbuffer,xr+xbuffer])
	
	ax.plot(np.real(w), np.imag(w), 'k-', linewidth=1.5, markersize=4.)
	ax.plot(np.real(w), np.imag(w), 'ko', linewidth=0, markersize=4.)
	
	# Remove top and right axes lines ("spines")
	spines_to_remove = ['top', 'right']
	for spine in spines_to_remove:
	    ax.spines[spine].set_visible(False)
		
	# Get rid of ticks. The position of the numbers is informative enough of
	# the position of the value.
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	
	# For remaining spines, thin out their line and change the black to a slightly off-black dark grey
	almost_black = '#262626'
	spines_to_keep = ['bottom', 'left']
	for spine in spines_to_keep:
	    ax.spines[spine].set_linewidth(0.5)
	    ax.spines[spine].set_color(almost_black)
		
	# Change the labels to the off-black
	ax.xaxis.label.set_color(almost_black)
	ax.yaxis.label.set_color(almost_black)
	
	# Change the axis title to off-black
	ax.title.set_color(almost_black)
	
	# Remove the line around the legend box, and instead fill it with a light grey
	# Also only use one point for the scatterplot legend because the user will 
	# get the idea after just one, they don't need three.
	# light_grey = np.array([float(248)/float(255)]*3)
	# legend = ax.legend(frameon=True)
	# rect = legend.get_frame()
	# rect.set_facecolor(light_grey)
	# rect.set_linewidth(0.0)
	# 
	# # Change the legend label colors to almost black, too
	# texts = legend.texts
	# for t in texts:
	#     t.set_color(almost_black)
		
	ax.set_title(titlestring,fontsize=16)
	plt.xlabel('$\mathbb{R}$',fontsize=17)
	plt.ylabel('$i \mathbb{R}$',fontsize=17)
	fig.savefig(filestring+format)
	if Plot_B:
		# plt.show()
		os.system("open "+filestring+format)
		
	plt.clf()
	plt.close('all')
	return 



