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
import scipy.fft

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

def doRLDeconvolution2(datapath , psfdatapath , niter=0, qtprocessbar=None):
    '''Deconvolute data using the Richardson Lucy algorithm.
    This code is based on the source code for DeconvolutionLab2
    https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    
    This is not working well, beads are being duplicated in the final volume'''
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

    #From https://github.com/scipy/scipy/blob/803e52d7e82cfc027daa55426466da29bc303b5c/scipy/signal/signaltools.py#L385
    def _centered(arr, newshape):
        # Return the center newshape portion of the array.
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

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

        #Determine best shape to do the fft
        #based on fftconvolve() code at https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
        #Simply use the maximum in each dimension
        s1 = data_np_norm.shape
        s2 = psf_np_norm.shape
        bestshape = [max((s1[i], s2[i])) for i in range(data_np_norm.ndim)]
        fslice = tuple([slice(sz) for sz in bestshape])
        #It may be pssoble to accelerate the FFT by choosing higher shape values
        #Similar to what is used in scipy _freq_domain_conv()  next_fast_len()
        #but we are not using this trick here

        psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
        xn1 = np.array(data_np_norm) #initialize

        progr0.increment() #4

        for i in range(niter):
            xn=xn1

            #convolution xn (*) psf
            xn_fft = scipy.fft.rfftn(xn, bestshape)
            conv1_fft = np.multiply(xn_fft , psf_fft)
            #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
            Hx = scipy.fft.irfftn(conv1_fft,bestshape)
            #Fix size of the result so that convolution result has the same shape as data
            #This is needed otherwise it will fail to divide
            Hx = _centered(Hx, data_np_norm.shape)

            yhx = np.divide(data_np_norm,Hx) #Problem, Hx shape has one index higher in zz
            
            progr0.increment()

            #correlation of the result with psf (note that is not a convolution)
            yhx_fft = scipy.fft.rfftn(yhx, bestshape)
            htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
            #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
            htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)
            htyhx = _centered(htyhx, data_np_norm.shape)

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

def doRLDeconvolution3(datapath , psfdatapath , niter=0, qtprocessbar=None):
    '''Deconvolute data using the Richardson Lucy algorithm.
    This code is based on the source code for DeconvolutionLab2
    https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    
    Not working well, doubling beads'''
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

    #From https://github.com/scipy/scipy/blob/803e52d7e82cfc027daa55426466da29bc303b5c/scipy/signal/signaltools.py#L385
    def _centered(arr, newshape):
        # Return the center newshape portion of the array.
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

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

        bestshape = data_np.shape

        def convertAndNormalise(d0):
            d1= d0.astype('float32')
            vmax= d0.max()
            vmin=d0.min()
            d2 = (d1-vmin)/(vmax-vmin)
            return d2
        
        data_np_norm = convertAndNormalise(data_np)
        psf_np_norm = convertAndNormalise(_centered(psf_np, bestshape))

        progr0.increment() #3

        psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
        xn1 = np.array(data_np_norm) #initialize copy

        progr0.increment() #4

        for i in range(niter):
            xn=xn1

            #convolution xn (*) psf
            xn_fft = scipy.fft.rfftn(xn, bestshape)
            conv1_fft = np.multiply(xn_fft , psf_fft)
            #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
            Hx = scipy.fft.irfftn(conv1_fft,bestshape)

            yhx = np.divide(data_np_norm,Hx)
            
            progr0.increment()

            #correlation of the result with psf (note that is not a convolution)
            yhx_fft = scipy.fft.rfftn(yhx, bestshape)
            htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
            #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
            htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)

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

def doRLDeconvolution4(datapath , psfdatapath , niter=0, qtprocessbar=None):
    '''Deconvolute data using the Richardson Lucy algorithm.
    This code is based on the source code for DeconvolutionLab2
    https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    '''
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

    def _centerarray(arr, newshape):
        #Center current array to newshape.
        #If newshape has higher widths then put the array in the middle
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        
        startind_newarr=np.zeros(newshape.size, dtype='uint')
        endin_newarr = np.zeros(newshape.size, dtype='uint')
        startin_currarray = np.zeros(newshape.size, dtype='uint')
        endin_currarray = np.zeros(newshape.size, dtype='uint')


        for k in range(currshape.size):
            #Default if shape value is the same
            startin_currarray[k] = 0
            endin_currarray[k] = currshape[k]
            startind_newarr[k]=0
            endin_newarr[k] = newshape[k]

            if currshape[k] < newshape[k]:
                startin_currarray[k] = 0
                endin_currarray[k] = currshape[k]
                startind_newarr[k] = (newshape[k]-currshape[k]) // 2
                endin_newarr[k] = startind_newarr[k] + currshape[k]
            if currshape[k] > newshape[k]:
                #crop current shape
                startin_currarray[k] = (currshape[k]-newshape[k]) // 2
                endin_currarray[k] = startin_currarray[k] + newshape[k]
                startind_newarr[k] = 0
                endin_newarr[k] = newshape[k]
            
        slicecurrarray = [slice(startin_currarray[k], endin_currarray[k]) for k in range(startin_currarray.size)]
        slicenewarray = [slice(startind_newarr[k], endin_newarr[k]) for k in range(startind_newarr.size)]

        arr0 = arr[tuple(slicecurrarray)]

        newarray = np.zeros(newshape, dtype=arr.dtype)
        newarray[tuple(slicenewarray)] = arr0
        #print(newarray[32,512,512]) #Debug
        return newarray

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

        s1 = data_np.shape
        s2 = psf_np.shape

        bestshape0 = [max((s1[i], s2[i])) for i in range(data_np.ndim)]

        #Gets shape using next_fast_len tool
        bestshape = [scipy.fft.next_fast_len(bestshape0[s0], True) for s0 in range(len(bestshape0))]

        def convertAndNormalise(d0):
            d1= d0.astype('float32')
            vmax= d0.max()
            vmin=d0.min()
            d2 = (d1-vmin)/(vmax-vmin)
            return d2
        
        #Normalise
        data_np0 = _centerarray(data_np,bestshape)
        data_np_norm = convertAndNormalise(data_np0)
        psf_np0 = _centerarray(psf_np,bestshape)
        psf_np_norm = convertAndNormalise(psf_np0)

        progr0.increment() #3

        psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
        xn1 = np.array(data_np_norm) #initialize copy

        progr0.increment() #4

        for i in range(niter):
            xn=xn1

            #convolution xn (*) psf
            xn_fft = scipy.fft.rfftn(xn, bestshape)
            conv1_fft = np.multiply(xn_fft , psf_fft)
            #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
            Hx = scipy.fft.irfftn(conv1_fft,bestshape)

            yhx = np.divide(data_np_norm,Hx)
            
            progr0.increment()

            #correlation of the result with psf (note that is not a convolution)
            yhx_fft = scipy.fft.rfftn(yhx, bestshape)
            htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
            #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
            htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)

            xn1 = np.multiply(xn, htyhx)
            
            progr0.increment()

        #At the end xn1 should have the deconvoluted data

        #Crop data to original shape
        fslice = tuple([slice(sz) for sz in data_np.shape])
        xn1 = xn1[fslice]
        
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

def doRLDeconvolution5(datapath , psfdatapath , niter=0, qtprocessbar=None):
    '''Deconvolute data using the Richardson Lucy algorithm.
    This code is based on the source code for DeconvolutionLab2
    https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    '''
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

        bestshape = data_np.shape

        def convertAndNormalise(d0):
            d1= d0.astype('float32')
            vmax= d0.max()
            vmin=d0.min()
            d2 = (d1-vmin)/(vmax-vmin)
            return d2
        
        #Normalise
        data_np_norm = convertAndNormalise(data_np)
        psf_np_norm = convertAndNormalise(psf_np)

        progr0.increment() #3

        xn1 = np.array(data_np_norm) #initialize copy

        progr0.increment() #4

        for i in range(niter):
            xn=xn1

            Hx = scipy.signal.convolve(xn, psf_np,mode='same', method='fft')

            yhx = np.divide(data_np_norm,Hx)
            
            progr0.increment()

            #correlation of the result with psf (note that is not a convolution)
            htyhx = scipy.signal.correlate(yhx, psf_np, mode='same', method='fft')

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

