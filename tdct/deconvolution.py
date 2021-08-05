import sys
import os

import tifffile
if getattr(sys, 'frozen', False):
	# programm runs in a bundle (pyinstaller)
	execdir = sys._MEIPASS
else:
	execdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(execdir)

import numpy as np
from PyQt5 import QtWidgets
from . import clrmsg
from . import TDCT_debug

try:
	import tifffile as tf
except:
	sys.exit("Please install tifffile, e.g.: pip install tifffile")

debug = TDCT_debug.debug

import scipy.signal

def doRLDeconvolution(datapath , psfdatapath , niter=0, qtprocessbar=None):
	#Internal class to handle progress bar
	class progrBarHandle():
		def __init__(self, qtprocessbar, niter):
			self.iter=0
			self.niter = niter
			self.qtbar = qtprocessbar

			# if self.qtbar:
			# 	self.qtbar.value= self.iter/self.niter*100
			# 	QtWidgets.QApplication.processEvents()
			self.updateUI() #Not working
		def increment(self):
			self.iter+=1
			self.updateUI()
		def updateUI(self):
			if self.qtbar:
				self.qtbar.setValue(self.iter/self.niter*100)
				QtWidgets.QApplication.processEvents()
		def setmax(self):
			self.iter= self.niter
			self.updateUI()

	#Estimate progress iterations
	nProgrIter = 2*niter + 6 #Check if ok
	progr0 = progrBarHandle(qtprocessbar, nProgrIter) #Sets up

	#Read data
	if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
		if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
		progr0.increment() #1

		data_np = tf.imread(datapath)
		psf_np = tf.imread(psfdatapath)

		progr0.increment() #2

		#Normalise both data (and also converts)

		def convertAndNormalise(d0):
			d1= d0.astype('float32')
			vmax= d0.max()
			vmin=d0.min()
			d2 = (d1-vmin)/(vmax-vmin)
			return d2
		
		data_np_norm = convertAndNormalise(data_np)
		psf_np_norm = convertAndNormalise(psf_np)

		progr0.increment() #3

		#Code inspired by scikit-image Richardson-lucy
		#https://github.com/scikit-image/scikit-image/blob/main/skimage/restoration/deconvolution.py#L383-L443
		psf_t = np.flip(psf_np_norm)
		xn1 = np.full(data_np_norm.shape,0.5, dtype=data_np_norm.dtype)

		progr0.increment() #4
		
		if debug is True: print(clrmsg.DEBUG, "Begining iterative deconvolution niter=", niter)

		for i in range(niter):
			xn=xn1
			
			Hx = scipy.signal.convolve(xn,psf_np_norm, mode='same')
			yhx = np.divide(data_np_norm,Hx)

			progr0.increment()

			htyhx = scipy.signal.convolve(yhx,psf_t, mode='same')
			xn1 = np.multiply(xn, htyhx)

			progr0.increment()

		#At the end xn1 should have the deconvoluted data
		#Convert to uint8
		data_deconv_norm_256 = convertAndNormalise(xn1)*256
		data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
		
		progr0.increment() #5

		#Save data
		fpath,fname = os.path.split(datapath)
		fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
		tf.imsave(fname0 ,data_deconv_uint8)

		progr0.setmax()
 
		if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)


