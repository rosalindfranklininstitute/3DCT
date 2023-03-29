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
from vispy.util import keys

from superqt import QDoubleRangeSlider
import math

# QSplitter does not seem to be working

class Ui_MainWindow(QMainWindow):
    def __init__(self, data_3d=None, data_2d=None,corr_params=None):
        super().__init__()

        self.setWindowTitle("3D Correlation Viewer")
        self.resize(800, 500)

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
        
        self.setupVispyCorrelCanvas(canvas, data_3d, data_2d, corr_params)
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

        splitter.setSizes((90,10))
        windowlayout.addWidget(splitter)

        self.setCentralWidget(centralWidget)
        #self.show()

        #Events
        self.btnResetCam.clicked.connect(self.resetCamera)
        self.sliderVol.valueChanged.connect(self.sliderVolChanged)
        self.sliderIm.valueChanged.connect(self.sliderImChanged)

        canvas.events.mouse_move.connect(self.on_mouse_move)
        canvas.events.key_press.connect(self.on_key_press)

    def setupVispyCorrelCanvas(self, canvas, data_3d, data_2d, correl_params):      
        #Add content
        self.view = canvas.central_widget.add_view()

        #Image as grayscale
        # Create the visual for plane rendering
        self.image=vispy.scene.visuals.Image(
            data = data_2d,
            parent=self.view.scene,
            cmap='grays'
        )

        #Volume shown as green
        import vispy.color.colormap as cm
        mygreen_cm = cm.Colormap([(0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0) ],controls=None, interpolation='linear')

        self.volume = vispy.scene.visuals.Volume( data_3d,
            parent=self.view.scene,
            raycasting_mode='volume',
            method='mip',
            cmap=mygreen_cm
        )
        self.volume.opacity=1.0
        self.volume.set_gl_state('additive')

        #Initial rotation based in provided correlation values
        phi, theta, psi = correl_params['phi'], correl_params['theta'], correl_params['psi']
        #Does not store internally the angles, but the rotation matrix itself
        self.volRotationTransf = self.getRotMatrixTransformFromEuler(phi, theta, psi)

        self.scale = correl_params['scale']
        self.translate = correl_params['transl']

        #Set the all of tranformations 
        # rotation as defined above, scale and translation)
        self.setVolumeTransform()


        # volume.transform.scale((scale0,scale0,scale0))
        # volume.transform.translate()

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

    def getRotMatrixTransformFromEuler(self,phi,theta,psi):
        tr = MatrixTransform()
        tr.rotate(phi,(0,0,1)) # axis  (x,y,z)
        tr.rotate(theta,axis=(1,0,0))
        tr.rotate(psi,(0,0,1))
        return tr

    def setVolumeTransform(self):
        #Uses self.VolRotation to apply rotation
        self.volume.transform =  self.volRotationTransf

        self.volume.transform.scale((self.scale,self.scale,self.scale))
        self.volume.transform.translate(self.translate)


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
    
    def on_key_press(self,event):
        d=1
        #print(type(event.key))

        if keys.SHIFT in event.modifiers:
            d*=5
        if keys.CONTROL in event.modifiers:
            d/=5
        
        #Translation
        if event.key.name=='W':
            self.volume.transform.translate((0,-d,0))
        elif event.key.name=='S':
            self.volume.transform.translate((0,d,0))
        elif event.key.name=='A':
            self.volume.transform.translate((-d,0,0))
        elif event.key.name=='D':
            self.volume.transform.translate((d,0,0))
        #Scaling
        elif event.key.name=='Q':
            sc = (100-d)/100
            self.volume.transform.scale((sc,sc,sc))
        elif event.key.name=='E':
            sc = (100+d)/100
            self.volume.transform.scale((sc,sc,sc))

    def on_mouse_move(self,event):
        #print(f"Mouse moved, event buttons:{event.buttons} , modifiers:{event.modifiers}")
        if event.button==1 and event.is_dragging and len(event.modifiers)>0:
            #print(type(event.modifiers[0]))
            if keys.CONTROL in event.modifiers:
                dxy = event.pos - event.last_event.pos
                x,y = dxy/2

                v0 = math.sqrt(x*x+y*y)
                if v0>0:
                    rot_vector = (-y,x,0)
                    #Apply this dxy to rotation
                    # rotate around a perpendicular of displacement
                    self.volRotationTransf.rotate(v0,rot_vector)


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
        'theta':-45,
        'psi':0,
        'scale':1.0,
        'transl':[0,0,0]
    }
    main=showCorrelation(data_3D, data_2D, correl_params)

    # window=Ui_MainWindow()
    # window.show()

    app.exec()

    #sys.exit(app.exec_())
