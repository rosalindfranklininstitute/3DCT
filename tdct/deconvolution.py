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

import RedLionfishDeconv as rld

#import tkinter.messagebox

#Internal helper class to handle progress bar
class _progrBarHandle():
    def __init__(self, qtprocessbar, maxiter):
        self.iter=0
        self.maxiter = maxiter
        self.qtbar = qtprocessbar

        self.updateUI() #Not working
    def increment(self):
        self.iter+=1
        #If maximum value is reached, restart from zero
        #In block iteration algorithm, the progress bar only works as a busy indicator
        if self.iter>=self.maxiter:
            self.iter=0
        self.updateUI()
    def updateUI(self):
        if self.qtbar:
            self.qtbar.setValue(self.iter/self.maxiter*100)
            QtWidgets.QApplication.processEvents()
    def setmax(self):
        self.iter= self.maxiter
        self.updateUI()


def doRLDeconvolutionFromFiles(datapath , psfdatapath,niter=0, useGPU = True, qtprocessbar=None):
    #Sets up progress bar
    nProgrIter=niter+2
    progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

    if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
        if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)

        data_np = tf.imread(datapath)
        psf_np = tf.imread(psfdatapath)

        def _functCallbackTick():
            progr0.increment()
        
        data_deconv_uint8 = rld.doRLDeconvolutionFromNpArrays(data_np, psf_np, niter=niter, callbkTickFunc=_functCallbackTick,resAsUint8=True)

        if not data_deconv_uint8 is None:

            #Save data
            fpath,fname = os.path.split(datapath)
            rootfname, ext = os.path.splitext(fname)

            fname0 = os.path.join(fpath, rootfname+"_DeconvRL_it"+str(niter)+ ext)
            tf.imsave(fname0 ,data_deconv_uint8)

            progr0.setmax()

            if debug is True: print(clrmsg.DEBUG, "Completed RL deconvolution. file saved: ", fname0)
            
            #Doesnt work
            # QtWidgets.QMessageBox.about(progr0.parent(), f"Completed RL deconvolution. File save as {str(fname0)} .")
            #tkinter.messagebox.showinfo("RL Deconvolution completed", f"File save as {str(fname0)} .")

            QtWidgets.QMessageBox.about(None,"RL Deconvolution completed", f"Result saved in file\n{str(fname0)}" )

        else:
            #deconvolution failed
            if debug is True: print(clrmsg.DEBUG, "RL Deconvolution failed.")

            #tkinter.messagebox.showerror("Error", "RL Deconvolution failed.")
            QtWidgets.QMessageBox.about(None,"Error", "RL Deconvolution failed.")



        #No return needed




# def _centered(arr, newshape):
#     # Return the center newshape portion of the array.
#     newshape = np.asarray(newshape)
#     currshape = np.array(arr.shape)
#     startind = (currshape - newshape) // 2
#     endind = startind + newshape
#     myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
#     return arr[tuple(myslice)]

# def doRLDeconvolution(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     #Internal class to handle progress bar

#     ''' This code is not giving correct results
#     This is probably because the usage of the psf flipped
#     In a later version doRLDeconvolution5 (based on DeconvolutionLab2 instead of using psf-flipped in the 2nd convolution,
#     a correlation is performed. '''

#     #Estimate progress iterations
#     nProgrIter = 2*niter + 6 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
#         progr0.increment() #1

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #2

#         #Normalise both data (and also converts)

#         data_np_norm = _convertAndNormalise(data_np)
#         psf_np_norm = _convertAndNormalise(psf_np)

#         progr0.increment() #3

#         #Code inspired by scikit-image Richardson-lucy
#         #https://github.com/scikit-image/scikit-image/blob/main/skimage/restoration/deconvolution.py#L383-L443
#         psf_t = np.flip(psf_np_norm)
#         xn1 = np.full(data_np_norm.shape,0.5, dtype=data_np_norm.dtype)

#         progr0.increment() #4
        
#         if debug is True: print(clrmsg.DEBUG, "Begining iterative deconvolution niter=", niter)

#         for i in range(niter):
#             xn=xn1
            
#             Hx = scipy.signal.convolve(xn,psf_np_norm, mode='same')
#             yhx = np.divide(data_np_norm,Hx)

#             progr0.increment()

#             htyhx = scipy.signal.convolve(yhx,psf_t, mode='same')
#             xn1 = np.multiply(xn, htyhx)

#             progr0.increment()

#         #At the end xn1 should have the deconvoluted data
#         #Convert to uint8
#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #5

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)

# def doRLDeconvolution2(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''Deconvolute data using the Richardson Lucy algorithm.
#     This code is based on the source code for DeconvolutionLab2
#     https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    
#     This is not working well, beads are being duplicated in the final volume'''

#     #From https://github.com/scipy/scipy/blob/803e52d7e82cfc027daa55426466da29bc303b5c/scipy/signal/signaltools.py#L385
#     def _centered(arr, newshape):
#         # Return the center newshape portion of the array.
#         newshape = np.asarray(newshape)
#         currshape = np.array(arr.shape)
#         startind = (currshape - newshape) // 2
#         endind = startind + newshape
#         myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
#         return arr[tuple(myslice)]

#     #Estimate progress iterations
#     nProgrIter = 2*niter + 6 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
#         progr0.increment() #1

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #2

#         #Normalise both data (and also converts)
        
#         data_np_norm = _convertAndNormalise(data_np)
#         psf_np_norm = _convertAndNormalise(psf_np)

#         progr0.increment() #3

#         #Determine best shape to do the fft
#         #based on fftconvolve() code at https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py
#         #Simply use the maximum in each dimension
#         s1 = data_np_norm.shape
#         s2 = psf_np_norm.shape
#         bestshape = [max((s1[i], s2[i])) for i in range(data_np_norm.ndim)]
#         fslice = tuple([slice(sz) for sz in bestshape])
#         #It may be pssoble to accelerate the FFT by choosing higher shape values
#         #Similar to what is used in scipy _freq_domain_conv()  next_fast_len()
#         #but we are not using this trick here

#         psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
#         xn1 = np.array(data_np_norm) #initialize

#         progr0.increment() #4

#         for i in range(niter):
#             xn=xn1

#             #convolution xn (*) psf
#             xn_fft = scipy.fft.rfftn(xn, bestshape)
#             conv1_fft = np.multiply(xn_fft , psf_fft)
#             #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
#             Hx = scipy.fft.irfftn(conv1_fft,bestshape)
#             #Fix size of the result so that convolution result has the same shape as data
#             #This is needed otherwise it will fail to divide
#             Hx = _centered(Hx, data_np_norm.shape)

#             yhx = np.divide(data_np_norm,Hx) #Problem, Hx shape has one index higher in zz
            
#             progr0.increment()

#             #correlation of the result with psf (note that is not a convolution)
#             yhx_fft = scipy.fft.rfftn(yhx, bestshape)
#             htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
#             #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
#             htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)
#             htyhx = _centered(htyhx, data_np_norm.shape)

#             xn1 = np.multiply(xn, htyhx)
            
#             progr0.increment()

#         #At the end xn1 should have the deconvoluted data
        
#         #Convert to uint8
#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #5

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)

# def doRLDeconvolution3(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''Deconvolute data using the Richardson Lucy algorithm.
#     This code is based on the source code for DeconvolutionLab2
#     https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
    
#     Not working well, doubling beads'''

#     #From https://github.com/scipy/scipy/blob/803e52d7e82cfc027daa55426466da29bc303b5c/scipy/signal/signaltools.py#L385
#     def _centered(arr, newshape):
#         # Return the center newshape portion of the array.
#         newshape = np.asarray(newshape)
#         currshape = np.array(arr.shape)
#         startind = (currshape - newshape) // 2
#         endind = startind + newshape
#         myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
#         return arr[tuple(myslice)]

#     #Estimate progress iterations
#     nProgrIter = 2*niter + 6 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
#         progr0.increment() #1

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #2

#         #Normalise both data (and also converts)

#         bestshape = data_np.shape
        
#         data_np_norm = _convertAndNormalise(data_np)
#         psf_np_norm = _convertAndNormalise(_centered(psf_np, bestshape))

#         progr0.increment() #3

#         psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
#         xn1 = np.array(data_np_norm) #initialize copy

#         progr0.increment() #4

#         for i in range(niter):
#             xn=xn1

#             #convolution xn (*) psf
#             xn_fft = scipy.fft.rfftn(xn, bestshape)
#             conv1_fft = np.multiply(xn_fft , psf_fft)
#             #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
#             Hx = scipy.fft.irfftn(conv1_fft,bestshape)

#             yhx = np.divide(data_np_norm,Hx)
            
#             progr0.increment()

#             #correlation of the result with psf (note that is not a convolution)
#             yhx_fft = scipy.fft.rfftn(yhx, bestshape)
#             htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
#             #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
#             htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)

#             xn1 = np.multiply(xn, htyhx)
            
#             progr0.increment()

#         #At the end xn1 should have the deconvoluted data
        
#         #Convert to uint8
#         data_deconv_norm_256 = convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #5

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)

# def doRLDeconvolution4(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''Deconvolute data using the Richardson Lucy algorithm.
#     This code is based on the source code for DeconvolutionLab2
#     https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java

#     Not working
#     '''

#     def _centerarray(arr, newshape):
#         #Center current array to newshape.
#         #If newshape has higher widths then put the array in the middle
#         newshape = np.asarray(newshape)
#         currshape = np.array(arr.shape)
        
#         startind_newarr=np.zeros(newshape.size, dtype='uint')
#         endin_newarr = np.zeros(newshape.size, dtype='uint')
#         startin_currarray = np.zeros(newshape.size, dtype='uint')
#         endin_currarray = np.zeros(newshape.size, dtype='uint')


#         for k in range(currshape.size):
#             #Default if shape value is the same
#             startin_currarray[k] = 0
#             endin_currarray[k] = currshape[k]
#             startind_newarr[k]=0
#             endin_newarr[k] = newshape[k]

#             if currshape[k] < newshape[k]:
#                 startin_currarray[k] = 0
#                 endin_currarray[k] = currshape[k]
#                 startind_newarr[k] = (newshape[k]-currshape[k]) // 2
#                 endin_newarr[k] = startind_newarr[k] + currshape[k]
#             if currshape[k] > newshape[k]:
#                 #crop current shape
#                 startin_currarray[k] = (currshape[k]-newshape[k]) // 2
#                 endin_currarray[k] = startin_currarray[k] + newshape[k]
#                 startind_newarr[k] = 0
#                 endin_newarr[k] = newshape[k]
            
#         slicecurrarray = [slice(startin_currarray[k], endin_currarray[k]) for k in range(startin_currarray.size)]
#         slicenewarray = [slice(startind_newarr[k], endin_newarr[k]) for k in range(startind_newarr.size)]

#         arr0 = arr[tuple(slicecurrarray)]

#         newarray = np.zeros(newshape, dtype=arr.dtype)
#         newarray[tuple(slicenewarray)] = arr0
#         #print(newarray[32,512,512]) #Debug
#         return newarray

#     #Estimate progress iterations
#     nProgrIter = 2*niter + 6 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
#         progr0.increment() #1

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #2

#         s1 = data_np.shape
#         s2 = psf_np.shape

#         bestshape0 = [max((s1[i], s2[i])) for i in range(data_np.ndim)]

#         #Gets shape using next_fast_len tool
#         bestshape = [scipy.fft.next_fast_len(bestshape0[s0], True) for s0 in range(len(bestshape0))]
        
#         #Normalise
#         data_np0 = _centerarray(data_np,bestshape)
#         data_np_norm = _convertAndNormalise(data_np0)
#         psf_np0 = _centerarray(psf_np,bestshape)
#         psf_np_norm = _convertAndNormalise(psf_np0)

#         progr0.increment() #3

#         psf_fft = scipy.fft.rfftn(psf_np_norm, bestshape)
#         xn1 = np.array(data_np_norm) #initialize copy

#         progr0.increment() #4

#         for i in range(niter):
#             xn=xn1

#             #convolution xn (*) psf
#             xn_fft = scipy.fft.rfftn(xn, bestshape)
#             conv1_fft = np.multiply(xn_fft , psf_fft)
#             #Hx = scipy.fft.ifftn(conv1_fft,bestshape)
#             Hx = scipy.fft.irfftn(conv1_fft,bestshape)

#             yhx = np.divide(data_np_norm,Hx)
            
#             progr0.increment()

#             #correlation of the result with psf (note that is not a convolution)
#             yhx_fft = scipy.fft.rfftn(yhx, bestshape)
#             htyhx_fft = np.multiply(yhx_fft , np.conjugate(psf_fft))
#             #htyhx = scipy.fft.ifftn(htyhx_fft,bestshape)
#             htyhx = scipy.fft.irfftn(htyhx_fft,bestshape)

#             xn1 = np.multiply(xn, htyhx)
            
#             progr0.increment()

#         #At the end xn1 should have the deconvoluted data

#         #Crop data to original shape
#         fslice = tuple([slice(sz) for sz in data_np.shape])
#         xn1 = xn1[fslice]
        
#         #Convert to uint8
#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #5

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)

# def doRLDeconvolution5(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''Deconvolute data using the Richardson Lucy algorithm.
#     This code is based on the source code for DeconvolutionLab2
#     https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
#     Instead of doing convolutions using raw fft routines, Here we use convolution algorithms in scipy.signal.convolve and scipy.signal.correlate

#     This is working well and appears to give same results as DeconvolutionLab2
#     '''

#     #Estimate progress iterations
#     nProgrIter = 2*niter + 6 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)
#         progr0.increment() #1

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #2

#         bestshape = data_np.shape
        
#         #Normalise
#         data_np_norm = _convertAndNormalise(data_np)
#         psf_np_norm = _convertAndNormalise(psf_np)

#         progr0.increment() #3

#         xn1 = np.array(data_np_norm) #initialize copy

#         progr0.increment() #4

#         for i in range(niter):
#             xn=xn1

#             Hx = scipy.signal.convolve(xn, psf_np,mode='same', method='fft')

#             yhx = np.divide(data_np_norm,Hx)
            
#             progr0.increment()

#             #correlation of the result with psf (note that is not a convolution)
#             htyhx = scipy.signal.correlate(yhx, psf_np, mode='same', method='fft')

#             xn1 = np.multiply(xn, htyhx)
            
#             progr0.increment()

#         #At the end xn1 should have the deconvoluted data
        
#         #Convert to uint8
#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #5

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)


# def doRLDeconvolution7(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''RL deconvolution based in DeconvolutionLab2 with optional parameter for normalising inputs
#     Reversed engineered convolution and correlation for faster processing
#     https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/signaltools.py#L1293-L1413
#     for mode='same', method='fft', fftconvolution()
#     normaliseinputs set to false
#     '''

#     #no need to swap inputs in mode='same'
    
#     #Estimate progress iterations
#     nProgrIter = 2*niter + 5 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     normaliseinputs=False

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #1

#         #Convert and normalise
#         data_np_norm = _convertAndNormalise(data_np,normaliseinputs)
#         psf_np_norm = _convertAndNormalise(psf_np, normaliseinputs)

#         s1 = data_np_norm.shape
#         s2 = psf_np_norm.shape

#         shape = [(s1[i] + s2[i] - 1) for i in range(data_np_norm.ndim)]
#         fshape = [scipy.fft.next_fast_len(shape[a], True) for a in range(len(shape))]

#         #scipy.fft routines need to axes to be provided otherwise they only do FFT in last axes
#         #https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html
#         axes = [i for i in range(data_np_norm.ndim)]

#         progr0.increment() #2

#         #Precalculated psf_fft
#         psf_fft= scipy.fft.rfftn(psf_np_norm,fshape, axes)

#         #and precalculate psfconjugate
#         psf_reverse_sliceobj = (slice(None, None, -1),) * psf_np_norm.ndim
#         #This object is for reversing (mirror) an array in all the dimensions, z, y and x axis

#         #Reverses the psf and also calculates its conjugate
#         #This is perhaps unnecessary since the psf array is real type
#         psf_reverse_conj= psf_np_norm[psf_reverse_sliceobj].conj()
#         #Probably equivalent to np.flip with axis=None

#         psf_reverse_conj_fft = scipy.fft.rfftn( psf_reverse_conj,fshape, axes)

#         #Precalculate fslice (used to fix image sizes after using fast_len method)
#         fslice = tuple([slice(sz) for sz in shape])

#         progr0.increment() #3

#         xn1 = np.array(data_np_norm) #initialize copy
#         for i in range(niter):
#             xn=xn1

#             #Hx = scipy.signal.convolve(xn, psf_np,mode='same', method='fft')
#             xn_fft = scipy.fft.rfftn(xn,fshape, axes)
#             ret = scipy.fft.irfftn(xn_fft * psf_fft, fshape, axes)
#             #using fast_len, so fix sizes
#             #fslice = tuple([slice(sz) for sz in shape])
#             Hx0 = ret[fslice]
#             Hx = _centered(Hx0, s1).copy() #Fix shape as in mode='same'

#             yhx = np.divide(data_np_norm,Hx)

#             progr0.increment()

#             #correlation of the result with psf (note that is not a convolution)
#             #htyhx = scipy.signal.correlate(yhx, psf_np, mode='same', method='fft')
#             yhx_fft = scipy.fft.rfftn(yhx,fshape, axes)
#             ret = scipy.fft.irfftn(yhx_fft * psf_reverse_conj_fft, fshape, axes)
#             #using fast_len, so fix sizes
#             #fslice = tuple([slice(sz) for sz in shape])
#             htyhx0 = ret[fslice]
#             htyhx = _centered(htyhx0, s1).copy() #Fix shape as in mode='same'

#             xn1 = np.multiply(xn, htyhx)

#             progr0.increment()

#         data_deconv_norm_256 = _convertAndNormalise(xn1,True)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')
        
#         progr0.increment() #4

#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)


# def doRLDeconvolution10(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''RL deconvolution based on doRLDeconvolution7() that was working well
#     Reversed engineered convolution and correlation for faster processing
#     https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/signaltools.py#L1293-L1413
#     for mode='same', method='fft', fftconvolution() #
    
#     Simplify by using np.flip and no conjugation
#     Working well
#     '''
#     #Estimate progress iterations
#     nProgrIter = 2*niter + 5 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     normaliseinputs=False

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #1

#         #Convert and normalise
#         data_np_norm = _convertAndNormalise(data_np,normaliseinputs)
#         psf_np_norm = _convertAndNormalise(psf_np, normaliseinputs)

#         s1 = data_np_norm.shape
#         s2 = psf_np_norm.shape

#         shape = [(s1[i] + s2[i] - 1) for i in range(data_np_norm.ndim)]
#         fshape = [scipy.fft.next_fast_len(shape[a], True) for a in range(len(shape))]

#         #scipy.fft routines need to axes to be provided otherwise they only do FFT in last axes
#         #https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html
#         axes = [i for i in range(data_np_norm.ndim)]

#         progr0.increment() #2

#         #Precalculated psf_fft
#         psf_fft= scipy.fft.rfftn(psf_np_norm,fshape, axes)

#         #and precalculate psf_flip_fft
#         psf_flip = np.flip(psf_np_norm)
#         psf_flip_fft = scipy.fft.rfftn(psf_flip,fshape, axes)

#         #Precalculate fslice (used to fix image sizes after using fast_len method)
#         fslice = tuple([slice(sz) for sz in shape])

#         progr0.increment() #3

#         xn1 = np.array(data_np_norm) #initialize copy
#         for i in range(niter):
#             xn=xn1

#             #denominator convolution
#             xn_fft = scipy.fft.rfftn(xn,fshape, axes)
#             ret = scipy.fft.irfftn(xn_fft * psf_fft, fshape, axes)
#             #using fast_len, so fix sizes
#             Hx0 = ret[fslice]
#             Hx = _centered(Hx0, s1).copy() #Fix shape as in mode='same'

#             yhx = np.divide(data_np_norm,Hx)

#             progr0.increment()

#             #2nd convolution
#             yhx_fft = scipy.fft.rfftn(yhx,fshape, axes)
#             ret = scipy.fft.irfftn(yhx_fft * psf_flip_fft, fshape, axes)
#             #using fast_len, so fix sizes
#             htyhx0 = ret[fslice]
#             htyhx = _centered(htyhx0, s1).copy() #Fix shape as in mode='same'

#             xn1 = np.multiply(xn, htyhx)

#             progr0.increment()

#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')

#         progr0.increment() #4
        
#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)

#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)


# #An implementation of the circular() routine from DeconvolutionLab2 that shifts the data centre of the psf
# def _circulify3D(data3D):
#     #Check data is 3D
#     if data3D.ndim != 3:
#         raise Exception("Error, input data is not 3-dimensional")

#     data0 = np.array(data3D)

#     shape = data0.shape

#     zdimhalf0 = int(shape[0]/2)
#     zdimhalf1 = shape[0]-zdimhalf0

#     ydimhalf0 = int(shape[1]/2)
#     ydimhalf1 = shape[1]-ydimhalf0

#     xdimhalf0 = int(shape[2]/2)
#     xdimhalf1 = shape[2]-xdimhalf0

#     if zdimhalf0 >=1 and ydimhalf0>=1 and xdimhalf0>=1:
#         data0[zdimhalf1: , ydimhalf1:, xdimhalf1:] = data3D[:zdimhalf0 , :ydimhalf0 , :xdimhalf0 ] #cube 1
#         data0[zdimhalf1: , ydimhalf1:, :xdimhalf1] = data3D[:zdimhalf0 , :ydimhalf0 , xdimhalf0: ] #cube 2
#         data0[zdimhalf1: , :ydimhalf1, :xdimhalf1] = data3D[:zdimhalf0 , ydimhalf0: , xdimhalf0: ] #cube 3
#         data0[zdimhalf1: , :ydimhalf1, xdimhalf1:] = data3D[:zdimhalf0 , ydimhalf0: , :xdimhalf0 ] #cube 4
#         data0[:zdimhalf1 , ydimhalf1:, xdimhalf1:] = data3D[zdimhalf0: , :ydimhalf0 , :xdimhalf0 ] #cube 5
#         data0[:zdimhalf1 , ydimhalf1:, :xdimhalf1] = data3D[zdimhalf0: , :ydimhalf0 , xdimhalf0: ] #cube 6
#         data0[:zdimhalf1 , :ydimhalf1, :xdimhalf1] = data3D[zdimhalf0: , ydimhalf0: , xdimhalf0: ] #cube 7
#         data0[:zdimhalf1 , :ydimhalf1, xdimhalf1:] = data3D[zdimhalf0: , ydimhalf0: , :xdimhalf0 ] #cube 8
    
#     return data0

# def _change3DSizeAs(psf, data):
#     #based in size() code found in
#     #https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/e9af0aba493ba137d70877154648e5583e376a81/src/main/java/signal/RealSignal.java#L531
#     #Adapts psf shape  to the shape of data by padding/cropping

#     mz,my, mx = data.shape
#     nz,ny,nx = psf.shape

#     psf1 = np.zeros_like( data )
    
#     vx = min(nx,mx)
#     vy = min(ny,my)
#     vz = min(nz, mz)
#     ox = int((mx - nx) / 2)
#     oy = int((my - ny) / 2)
#     oz = int((mz - nz) / 2)

#     for k in range(vz):
#         for j in range(vy):
#             for i in range(vx):
#                 pi = i+ox if ox>=0 else i
#                 qi = i if ox>=0 else i-ox
#                 pj = j+oy if oy>=0 else j
#                 qj = j if oy>=0 else j-oy
#                 pk = k+oz if oz>=0 else k
#                 qk = k if oz>=0 else k-oz
#                 psf1[pk,pj,pi] = psf[qk,qj,qi]
                
#     return psf1

# def doRLDeconvolution_DL2_2(datapath , psfdatapath , niter=0, qtprocessbar=None):
#     '''
#     RL deconvolution based in DeconvolutionLab2 with optional parameter for normalising inputs
#     Mimics DeconvolutionLab2 (DL2) as best as possible
#     https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucy.java
#     However it prepares the psf to have the same size as the input data
#     In DeconvolutionLab2 only psf is normalised but here both image data and psf are normalised
#     '''
#     #Copy of DeconvolutionLab2 (DL2)
#     #no need to swap inputs in mode='same'
    
#     #Estimate progress iterations
#     nProgrIter = 2*niter + 5 #Check if ok
#     progr0 = _progrBarHandle(qtprocessbar, nProgrIter) #Sets up

#     normaliseinputs=False

#     #Read data
#     if os.path.isfile(datapath) is True and os.path.isfile(psfdatapath) is True and niter>0:
#         if debug is True: print(clrmsg.DEBUG, "Loading images: ", datapath," , ",psfdatapath)

#         data_np = tf.imread(datapath)
#         psf_np = tf.imread(psfdatapath)

#         progr0.increment() #1    

#         #Convert and normalise
#         data_np_norm =_convertAndNormalise(data_np,False) #don't normalise
        
#         #Normalise to the sum (not to maximum)
#         psf0 = psf_np.astype('float32')
#         sum = np.sum(psf0)
#         if sum==0:
#             sum=1.0
#         psf_norm = psf0/sum

#         #Adapts psf shape to data
#         psf0 = _change3DSizeAs(psf_norm, data_np_norm)
#         #psf1 should have the same dimensions as data

#         #Do circulirisation of psf data before starting the RL algorithm
#         psf1 = _circulify3D(psf0)

#         progr0.increment() #2

#         #Precalculated psf_fft
#         psf_fft= scipy.fft.rfftn(psf1)
#         psf_fft_conj = np.conjugate(psf_fft) #conjugate

#         progr0.increment() #3

#         xn1 = np.array(data_np_norm) #initialize copy
#         for i in range(niter):
#             xn=xn1
#             xn_fft = scipy.fft.rfftn(xn)  
#             U0 = np.multiply(xn_fft * psf_fft)
#             u0 = scipy.fft.irfftn(U0)

#             p = np.divide(data_np_norm,u0)

#             progr0.increment()

#             U1 = scipy.fft.rfftn(p)

#             U2 = np.multiply(U1 * psf_fft_conj)

#             u2 = scipy.fft.irfftn(U2)

#             xn1 = np.multiply(xn, u2)

#             progr0.increment()

#         data_deconv_norm_256 = _convertAndNormalise(xn1)*256
#         data_deconv_uint8 = data_deconv_norm_256.astype('uint8')

#         progr0.increment() #4
        
#         #Save data
#         fpath,fname = os.path.split(datapath)
#         fname0 = os.path.join(fpath, "DeconvRL"+str(niter)+"it_"+fname)
#         tf.imsave(fname0 ,data_deconv_uint8)
        
#         progr0.setmax()
 
#         if debug is True: print(clrmsg.DEBUG, "Completed deconvolution, file saved: ", fname0)
