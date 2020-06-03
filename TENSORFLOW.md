# Compilation of Tensorflow C library
You might want to compile your own Tensorflow C library to make bodypix network
run faster. Tensorflow.js uses the Tensorflow C library. If you want to compile
your own Tensorflow C library, please following the steps described below. 

Install Go, if you are on Debian Buster, you can type in the following:

    sudo apt-get install golang


Set up ``GOPATH`` by following the instructions at 
[Go wiki](https://github.com/golang/go/wiki/SettingGOPATH#bash).

Install Bazelisk

    go get github.com/bazelbuild/bazelisk

Basically Tensorflow require Bazel as the build system. There are multiple
versions of Bazel which are not compatible to each other. The easiest way to 
launch Bazel is to use Bazelisk. 

Set up the environmental variable which specifies the version of Bazel used for
compiling Tensorflow

    export USE_BAZEL_VERSION=0.26.1


Clone Tensorflow repository

    git clone https://github.com/tensorflow/tensorflow.git


Change into Tensorflow directory

    cd tensorflow


Checkout the ``v1.15.0`` branch

    git checkout v1.15.0

Run the interactive configuration tool:

    ./configure

These are what I used in my configuration:

    $ ./configure 
    WARNING: --batch mode is deprecated. Please instead explicitly shut down 
    your Bazel server using the command "bazel shutdown".
    You have bazel 0.26.1 installed.
    Please specify the location of python. [Default is /usr/bin/python]: 
        /usr/bin/python3


    Found possible Python library paths:
    /usr/local/lib/python3.7/dist-packages
    /usr/lib/python3.7/dist-packages
    /usr/lib/python3/dist-packages
    Please input the desired Python library path to use.  Default is 
    [/usr/local/lib/python3.7/dist-packages]

    Do you wish to build TensorFlow with XLA JIT support? [Y/n]: y
    XLA JIT support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
    No OpenCL SYCL support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with ROCm support? [y/N]: n
    No ROCm support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with CUDA support? [y/N]: n
    No CUDA support will be enabled for TensorFlow.

    Do you wish to download a fresh release of clang? (Experimental) [y/N]: y
    Clang will be downloaded and used to compile tensorflow.

    Do you wish to build TensorFlow with MPI support? [y/N]: n
    No MPI support will be enabled for TensorFlow.

    Please specify optimization flags to use during compilation when bazel 
    option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:


    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
    Not configuring the WORKSPACE for Android builds.

    Preconfigured Bazel build configs. You can use any of the below by adding 
    "--config=<>" to your build command. See .bazelrc for more details.
            --config=mkl            # Build with MKL support.
            --config=monolithic     # Config for mostly static monolithic build.
            --config=gdr            # Build with GDR support.
            --config=verbs          # Build with libverbs support.
            --config=ngraph         # Build with Intel nGraph support.
            --config=numa           # Build with NUMA support.
            --config=dynamic_kernels        # (Experimental) Build kernels into 
    separate shared objects.
            --config=v2             # Build TensorFlow 2.x instead of 1.x.
    Preconfigured Bazel build configs to DISABLE default on features:
            --config=noaws          # Disable AWS S3 filesystem support.
            --config=nogcp          # Disable GCP support.
            --config=nohdfs         # Disable HDFS support.
            --config=noignite       # Disable Apache Ignite support.
            --config=nokafka        # Disable Apache Kafka support.
            --config=nonccl         # Disable NVIDIA NCCL support.
    Configuration finished

In case you wonder, 
- XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
algebra that can accelerate TensorFlow models with potentially no source code 
changes.
- ROCm stands for RadeonOpenCompute
- SYCL is a higher-level programming model for OpenCL
- MPI is message passing interface, which is needed if you want to run 
Tensorflow on a cluster. 

On my system, Tensorflow wouldn't compile unless I configure it to download 
Clang. 

If you are worried about the compiler optimisation flag, please have a look at
[GCC manual](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html), 
I know it is different to Clang, but they work in a similar way. The default
optimisation flags work fine. 

Build the Tensorflow C library by invoking

    bazel build --config opt //tensorflow/tools/lib_package:libtensorflow

Go to ``tensorflow/bazel-bin/tensorflow/`` directory, copy 
``libtensorflow_framework.so.1.15.0`` and ``libtensorflow.so.1.15.0`` to 
``bodypix/node_modules/@tensorflow/tfjs-node/deps/lib``. Please do note that
the target files already exist, and the write permission is unset. So you will
have to delete the target file first, before replacing them. You can run the
following commands.

    rm bodypix/node_modules/@tensorflow/tfjs-node/deps/lib/libtensorflow.so.1.15.0
    rm bodypix/node_modules/@tensorflow/tfjs-node/deps/lib/libtensorflow_framework.so.1.15.0
    cp tensorflow/bazel-bin/tensorflow/libtensorflow.so.1.15.0 bodypix/node_modules/@tensorflow/tfjs-node/deps/lib/
    cp tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.1.15.0 bodypix/node_modules/@tensorflow/tfjs-node/deps/lib/

Hopefully the bodypix network is now a bit faster for you.

## References
- https://github.com/golang/go/wiki/SettingGOPATH#bash
- https://github.com/bazelbuild/bazelisk
- https://www.tensorflow.org/install/source#configure_the_build
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md
- https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
