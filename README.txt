Encog C/C++ v1.0

This is the C/C++ version of Encog (http://www.heatonresearch.com/encog).
This file includes the complete source code for Encog for C.  The header
files are designed so that Encog can also be used with C++.  This file
includes instructions on how to compile and execute Encog for C.

** Visual C++ **

Just open the encog-c.sln file and compile as you would any Visual Studio 
project.

** UNIX **

Simply execute the make command in the directory that includes the 
Encog makefile. The makefile has been tested with Linux, MAC, and
Raspberry PI's Debian 7 release.

There are several options you can use.

To force 32 or 64 bit compile.

make ARCH=32
make ARCH=64

To compile with CUDA (for GPU).

make CUDA=1

You can also combine:

make ARCH=64 CUDA=1

Clear previous builds:

make clean

** Raspberry PI **

The gcc that comes with Raspberry PI seems to have trouble with the -m32 
option.  The following command will compile Encog for Raspberry PI.

make ARCH=RPI

For more information, visit:

http://www.heatonresearch.com/wiki/Encog_for_C

For binary releases, visit:

https://github.com/encog/encog-c/downloads
