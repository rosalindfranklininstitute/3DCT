# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Correl_VisPy_Widget.ui'
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import numpy as np

import vispy.app
import vispy.scene
#from vispy import app, scene
from  vispy.visuals.transforms.linear import MatrixTransform
from vispy.util.quaternion import Quaternion

from superqt import QDoubleRangeSlider

# QSplitter does not seem to be working

class Ui_MainWindow(QMainWindow):
    def __init__(self, data_3d=None, data_2d=None,corr_params=None):
        super().__init__()

        self.setWindowTitle("3D Correlation Viewer")
        self.resize(800, 600)

        #OK
        # centralWidget= QWidget(self)
        # windowlayout = QHBoxLayout(centralWidget)
        # #centralWidget.setLayout(windowlayout)
        # frameleft=QWidget(centralWidget)
        # frameleft.setStyleSheet('background-color:yellow')
        # #frameleft.setMinimumSize(200,200)
        # frameright=QWidget(centralWidget)
        # frameright.setStyleSheet('background-color:blue')
        # #frameright.setMinimumSize(200,200)
        # windowlayout.addWidget(frameleft)
        # windowlayout.addWidget(frameright)
        # self.setCentralWidget(centralWidget)
        # self.show()

        centralWidget= QWidget(self)
        windowlayout = QHBoxLayout(centralWidget)

        splitter=QSplitter(centralWidget)
        splitter.setOrientation(Qt.Horizontal)
        #splitter.setMinimumSize(512,512)

        frameleft=QWidget(splitter)
        splitter.addWidget(frameleft)

        #### setup VispyCanvas
        canvasl_Lt = QHBoxLayout(centralWidget)
        frameleft.setLayout(canvasl_Lt)
        
        #canvas = vispy.app.Canvas()
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        canvasl_Lt.addWidget(canvas.native)
        
        self.view, self.volume, self.image = self.setupVispyCorrelCanvas(canvas, data_3d, data_2d, corr_params)
        self.resetCamera()

        # self.volume.clim
        # self.image.clim

        frameright=QWidget(splitter)
        #frameright.setBaseSize(300,300)
        frameright.setMaximumWidth(300)

        #TODO add sliders
        # label sliderm, in form fashion
        formLayout = QFormLayout(centralWidget)
        self.sliderVol = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        formLayout.addRow(QLabel("volume"),self.sliderVol)
        self.sliderIm = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        formLayout.addRow(QLabel("image"),self.sliderIm)

        self.btnResetCam=QPushButton("Reset Camera", centralWidget)
        formLayout.addRow(self.btnResetCam, None)
        self.resetSliders() #Set sliders to max/min of data

        frameright.setLayout(formLayout)
        
        splitter.addWidget(frameright)
        #frameright.setMinimumSize(200,200)

        splitter.addWidget(frameleft)
        splitter.addWidget(frameright)

        splitter.setSizes((80,20))
        windowlayout.addWidget(splitter)

        self.setCentralWidget(centralWidget)
        #self.show()

        #Events
        self.btnResetCam.clicked.connect(self.resetCamera)
        self.sliderVol.valueChanged.connect(self.sliderVolChanged)
        self.sliderIm.valueChanged.connect(self.sliderImChanged)

    def setupVispyCorrelCanvas(self, canvas, data_3d, data_2d, correl_params):      
        #Add content
        view = canvas.central_widget.add_view()

        #Image as grayscale
        # Create the visual for plane rendering
        image=vispy.scene.visuals.Image(
            data = data_2d,
            parent=view.scene,
            cmap='grays'
        )

        #Volume shown as green
        import vispy.color.colormap as cm
        mygreen_cm = cm.Colormap([(0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0) ],controls=None, interpolation='linear')

        volume = vispy.scene.visuals.Volume( data_3d,
            parent=view.scene,
            raycasting_mode='volume',
            method='mip',
            cmap=mygreen_cm
        )
        volume.opacity=1.0
        volume.set_gl_state('additive')

        #Rotate for to the correlation parameters
        volume.transform= MatrixTransform()
        volume.transform.rotate(correl_params['phi'],(0,0,1)) # axis  (x,y,z)
        volume.transform.rotate(correl_params['theta'],axis=(1,0,0))
        volume.transform.rotate(correl_params['psi'],(0,0,1))#
        scale0 = correl_params['scale']
        volume.transform.scale((scale0,scale0,scale0))
        volume.transform.translate(correl_params['transl'])

        # Create a camera
        # cam = vispy.scene.cameras.ArcballCamera(
        #     fov=0, #orthographic
        #     interactive=True
        # )
        #Need to correct camera to point in zz direction
        #Default ArcballCamera points along y-axis for some reason
        # from vispy.util.quaternion import Quaternion
        # cam._quaternion = Quaternion.create_from_axis_angle(np.pi/2, 1,0,0) #OK

        # view.camera = cam

        return view, volume, image

    def resetSliders(self):
        #Sets to the current limits (clim) which is assumed to be max min of data
        vol_clim= self.volume.clim
        im_clim = self.volume.clim

        #print(vol_clim)
        #print(im_clim)

        #set ranges
        self.sliderVol.setRange(*vol_clim)
        self.sliderIm.setRange(*im_clim) 

        #set limits
        self.sliderVol.setValue(vol_clim)
        self.sliderIm.setValue(im_clim)
    
    def resetCamera(self):
        # Create a (new) camera
        cam0 = vispy.scene.cameras.ArcballCamera(
            fov=0, #orthographic
            parent=self.view.scene,
        )
        #Need to correct camera to point in zz direction
        #Default ArcballCamera points along y-axis for some reason
        cam0._quaternion = Quaternion.create_from_axis_angle(np.pi/2, 1,0,0) #OK
        #cam.scale_factor=1 #Need this otherwise reset() does not work
        #cam.set_default_state() #Does not work as a way to reset camera later
        self.view.camera=cam0
    
    def sliderVolChanged(self, newvals):
        #Sets new values
        self.volume.clim=list(newvals)

    def sliderImChanged(self, newvals):
        #Sets new values
        self.image.clim=list(newvals)

#Class instead of being a function because as soon as a function ends, it destroys the object created
class showCorrelation():
    def __init__(self,data_3d, data_2d, correl_params):
        self.window=Ui_MainWindow(data_3d, data_2d, correl_params)
        self.window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # # #Create mock data
    data_3D = np.random.rand(10,20,40)
    data_2D = np.random.rand(30,30)

    correl_params={
        'phi':0,
        'theta':45,
        'psi':0,
        'scale':1.0,
        'transl':[0,0,0]
    }
    main=showCorrelation(data_3D, data_2D, correl_params)

    # window=Ui_MainWindow()
    # window.show()

    app.exec()

    #sys.exit(app.exec_())
