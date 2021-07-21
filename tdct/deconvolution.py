import sys
import os
if getattr(sys, 'frozen', False):
	# programm runs in a bundle (pyinstaller)
	execdir = sys._MEIPASS
else:
	execdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(execdir)

import tifffile as tf
from PyQt5 import QtWidgets
from . import clrmsg
from . import TDCT_debug

import pycudadecon
