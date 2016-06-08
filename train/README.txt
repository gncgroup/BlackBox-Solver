(c) Victor Shevchenko, 2016
Tested only on *NIX.

Installation:
1. Install Shark-3.0.0 Library:
http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/getting_started/installation.html
2. Install python3, cython, pip3, numpy
3. Download CyMLP from the repository:
https://github.com/gncgroup/CyMLP
4. Build CyMLP with the following command:
python3 setup.py build_ext --inplace
5. Copy *.so, *.pxd or *.pyx files from CyMLP to this (train) and nn_projection folders
6. Build nn_projection, the (train) directory code:
python3 setup.py build_ext --inplace
7. Change directory to "controller" and build controller-module with the following commands:
cmake .
make
8. Run controller in background:
./EVO > /dev/null &
9. Change directory to ../weights/test (all weights-files will be saved here) and watch the training progress with ls command

You also can run controller in foreground:
./EVO
It will display the values of the objective function