![Encog Machine Learning Framework](http://www.heatonresearch.com/images/encog128.png)

Encog C/C++ v1.0 (experimental)
===============================

This is quick and experimental port of [Encog](http://www.encog.org) that I did for C/C++.  **I am not currently developing this port, but I am putting it on GitHub incase it is useful to someone.**  The primary purpose of this port was to experiment with CUDA.  However, it will work either with or without CUDA.  The CPU version does make use of [OpenMP](http://www.openmp.org/) for efficient processing.

This file includes the complete source code for Encog for C.  The header files are designed so that Encog can also be used with C++.  This file includes instructions on how to compile and execute Encog for C.

Visual C++
----------

Just open the encog-c.sln file and compile as you would any Visual Studio
project.

UNIX
----

Simply execute the make command in the directory that includes the
Encog makefile. The makefile has been tested with Linux, MAC, and
Raspberry PI's Debian 7 release.

There are several options you can use.

To force 32 or 64 bit compile.

```
make ARCH=32
make ARCH=64
```

To compile with CUDA (for GPU).

```
make CUDA=1
```

You can also combine:

```
make ARCH=64 CUDA=1
```

Clear previous builds:

```
make clean
```

Raspberry PI
------------

The gcc that comes with Raspberry PI seems to have trouble with the -m32
option.  The following command will compile Encog for Raspberry PI.

```
make ARCH=RPI
```

Encog CUDA Support
==================

Encog for C can make use of a nVidia CUDA enabled GPU for increased performance. Even if
you do not plan to program in C, you can use the Encog for C command line tool to train
neural networks. Encog for C makes use of the same EG Files and EGB Files used by other
Encog platforms, such as the Encog Workbench. CUDA is a very specialized architecture and
will not provide a performance boost for all operations. Currently CUDA can only be used
with the PSO training method. It is unlikely that RPROP will be extended to CUDA as the
CUDA architecture is not particularly conducive to RPROP. RPROP, due to is "backward
propagation" nature requires the activations of all neurons to be kept. Memory access is
one of the most cycle-intensive aspects of GPU programming. CUDA can achieve great speeds
when a SMALL amount of memory must be kept during training. CUDA also works well if a small
amount of memory is kept temporarily and then overwritten as training progresses. This is
the case with PSO.

Using CUDA with Encog for C
---------------------------

When Encog for C is compiled CUDA must be specified. The command to compile Encog with
CUDA is given here.

<pre>make CUDA=1 ARCH=64</pre>
The above command will compile Encog for CUDA and 64-bit CPU. This is the most advanced
build of Encog for C. I provide CUDA binaries for both Mac and Windows.
To find out if your version of Encog for C supports CUDA issue the following command.
<pre>encog-cmd CUDA</pre>
This will perform a simple test of the CUDA system. If you are using a CUDA Encog build
the version will be reported like this:
<pre>

* * Encog C/C++ (64 bit, CUDA) Command Line v1.0 * *

</pre>
If you are using a CUDA build, but your system does not have CUDA drivers or a CUDA GPU,
you will receive a system dependent error message. For more information, see the
troubleshooting section of Encog for C.

The CUDA build of Encog will always use the GPU if the training method supports it. To
disable the GPU, use the option /gpu:0. You can also specify /gpu:1 to enable the GPU;
however, this is redundant, given that the default operation is to use the GPU. The GPU
will only be used with PSO training.

A Simple Benchmark
------------------

The Encog command line utility contains a simple benchmark. This benchmark can be used to compare training results between GPU/CPU and CPU only. When the GPU is enabled, Encog is still making full use of your CPU cores. The GPU is simply brought in to assist with certain calculations. The following shows the output from a simple benchmark run. The benchmark is 10,000 data items of 10 inputs and one output, and 100 iterations of PSO. The following time is achieved using GPU and CPU.

<pre>heaton:encog-c jheaton$ ./encog benchmark /gpu:1

* * Encog C/C++ (64 bit, CUDA) Command Line v1.0 * *
Processor/Core Count: 8
Basic Data Type: double (64 bits)
GPU: enabled
Input Count: 10
Ideal Count: 1
Records: 10000
Iterations: 100

Performing benchmark...please wait
Benchmark time(seconds): 4.2172
Benchmark time includes only training time.

Encog Finished.  Run time 00:00:04.4040
heaton:encog-c jheaton$
As you can see from above, the benchmark was completed in 4.2 seconds. Now we will try the same benchmark, but disable the GPU.
heaton:encog-c jheaton$ ./encog benchmark /gpu:0

* * Encog C/C++ (64 bit, CUDA) Command Line v1.0 * *
Processor/Core Count: 8
Basic Data Type: double (64 bits)
GPU: disabled
Input Count: 10
Ideal Count: 1
Records: 10000
Iterations: 100

Performing benchmark...please wait
Benchmark time(seconds): 5.3727
Benchmark time includes only training time.

Encog Finished.  Run time 00:00:05.3749
heaton:encog-c jheaton$ </pre>

As you can see, the benchmark was completed in one less second.
As you increase the amount of training data the gap tends to increase.
On small training sets, the overhead of involving the GPU may actually slow training.
You would not want to use the GPU on a simple XOR train.

The above benchmark was performed on a MacBook Pro with an Intel i7 CPU and a nVidia
650M GPU. For more information on the computer see the article on Jeff's Computers.
Results will be better with more advanced GPU's. The M on the 650 also means that this is
a "mobile" edition of the GPU. Mobile GPU's tend to perform worse than desktop GPUs.
