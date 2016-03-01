from __future__ import print_function
import sys
import numpy as np



def ntrp6h(tint,t,y,tnew,ynew,yp,ypnew,Fmid,Spxint_requested=False):
# %NTRP6H  Interpolation helper function for BVP6C.
# %	  YINT = NTRP6H(TINT,T,Y,TNEW,YNEW,YP,YPNEW,FMID) evaluates the
# %	  Cash-Moore, Cash-Singhal6 based linear interpolant at time TINT.
# %	  TINT may be a scalar or a row vector.
# %	  [YINT,YPINT] = NTRP6H(TINT,T,Y,TNEW,YNEW,YP,YPNEW,FMID) returns
# %	  also the derivative of the interpolating polynomial.
# %
# %	  See also BVP6C, DEVAL, NTRP6C
#
# %	   Nick Hale  Imperial College London
# %	   $Date: 12/06/2006 $


# def interp_Hermite(w,h,y,yp,yp_ip025,yp_ip05,yp_ip075,both=True):
	#INTERP_HERMITE	 use the 6th order Hermite Interpolant presented by Cash
		#and Wright to find y and y' at abscissas for Quadrature.
	# As written, this requires most of its arguments to be 2d numpy arrays, and it returns 2d numpy arrays: (n,1)
	def A66(w):
		return w**2.*np.polyval(np.array([-24, 60, -50, 15.]),w)	  # w^2*(15-50*w+60*w^2-24*w^3);

	def B66(w):
		return w**2.*np.polyval(np.array([12, -26, 19, -5.])/3.,w)	   # w^2/3*(w-1)*(12*w^2-14*w+5);

	def C66(w):
		return w**2.*np.polyval(np.array([-8, 16, -8.])/3.,w)		   # -w^2*8/3*(1-w)^2;

	def D66(w):
		return w**2.*np.polyval(np.array([16, -40, 32, -8.]),w)	  # w^2*8*(1-w)^2*(2*w-1);


	def Ap66(w):
		return w*np.polyval(np.array([-120, 240, -150, 30.]),w)	 #w*(30-150*w+240*w^2-120*w^3);

	def Bp66(w):
		return w*np.polyval(np.array([20, -104/3., 19, -10/3.]),w)	  #w*(w*(20*w^2+19)-(104*w^2+10)/3);

	def Cp66(w):
		return -16/3.*w*np.polyval(np.array([2, -3, 1.]),w)		  #-16/3*w*(1-3*w+2*w^2);

	def Dp66(w):
		return w*np.polyval(np.array([80, -160, 96, -16.]),w)		 #w*(80*w^3-160*w^2+96*w-16);

	# print("------------------------------------------")
	# print("-------------Entering ntrp6y--------------")
	# print("------------------------------------------")

	# tint,t,y,tnew,ynew,yp,ypnew,Fmid - Input arguments
	# print("tint =",tint); print("t =",t); print("y =",y)
	# print("tnew =",tnew); print("ynew =",ynew)
	# print("yp =",yp); print("ypnew =",ypnew)
	# print('Fmid =',Fmid)
	h = tnew - t; w = (tint - t)/h;
	yint = np.zeros((ynew.size,len(tint)))
	# print("yint.shape =",yint.shape)
	
	for i in xrange(len(tint)):
		yint[:,i] = A66(w[i])*ynew + A66(1-w[i])*y	 + \
				( B66(w[i])*ypnew - B66(1-w[i])*yp + \
				  C66(w[i])*(Fmid[:,2]-Fmid[:,0]) + D66(w[i])*Fmid[:,1] )*h 
		if Spxint_requested:
			ypout[:,i] =( Ap66(w[i])*ynew  - Ap66(1-w[i])*y )/h + \
					 ( Bp66(w[i])*ypnew + Bp66(1-w[i])*yp + \
						 Cp66(w[i])*(Fmid[:,2]-Fmid[:,0]) + Dp66(w[i])*Fmid[:,1] )
	if not Spxint_requested: ypout = []
	
	# print("yint =",yint)
	# print("------------------------------------------")
	# print("--------------Leaving ntrp6y--------------")
	# print("------------------------------------------")
	return yint, ypout

	
#---------------------------------------------------------------------------
