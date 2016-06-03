## Synopsis

This application demonstrates the license plates recognition service. The application is designed to recognize license plates of the Republic of Kazakhstan. When you run the application, you can choose the **plates** folder with command **Open** in which the different sets of license plates, when you press **Run**, happening recognition of the license plates images.
To run demonstration application, you can run a script **main_window.py** or use or **Demo.bat**

This application is used for character recognition: Method of simple patterns, Support Vector Machine, Method k-Nearest neighbors, Artificial neural network and an Complex OCR method. To test the script recognition methods designed **test.py**, also you can use **Test OCR.bat**

## Installation

###1 Installing Python & Opencv & Numpy

####1.1 Python

Download Windows x86 MSI installer from:

https://www.python.org/downloads/release/python-2711/

#####Install:

Run installer and have a fun.

#####PATH:</br>
Add **"C:\Python27"** to **PATH** variable.

####1.2 Python package installer (pip)
Download from:</br>
https://bootstrap.pypa.io/get-pip.py

#####Install:
Open cmd, go to folder with **get-pip.py** and write:
> \> python get-pip.py 

#####PATH:</br>
Add **"C:\Python27\Scripts"** to **PATH** variable

####1.3 Numpy

#####Install:
Open cmd and type:
>\> pip install numpy

####1.4 Opencv

Download from:
https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.1.0/</br>
Extract OpenCV (will be extracted to a folder opencv)</br>
Copy ..\opencv\build\python\x86\2.7\\**cv2.pyd**</br>
Paste it in C:\Python27\Lib\site-packages</br>
Open Python IDLE or python console, and type:</br>
> \>\>\> import cv2</br>

If no errors shown, it is OK.</br>

###2 Installing FANN and PyQt4

####2.1 Wheel

Install **wheel** using command line:
> \> pip install wheel 

####2.2 Fann

#####Download:
Go to site: http://www.lfd.uci.edu/~gohlke/pythonlibs/</br>
Choose: fann2-1.0.7-cp27-none-win32.whl

#####Install:
Go to downloads folder.
Open cmd an type:
> \> pip install fann2-1.0.7-cp27-none-win32.whl

####2.3 Installing PyQt4 library

Go to: https://riverbankcomputing.com/software/pyqt/download/</br>
Download: PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe

OR just:</br>
http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe</br></br>

## Contributors

Contributor: **capital-boss**</br>
VK: https://vk.com/programmer22222</br>
Skype: programmer222222</br>

## License

This software licensed under the Apache Licence v2.0

