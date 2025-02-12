# Habana Gaudi2 kernel for llm.c
This repository provides the TPC kernels for [llm.c](https://github.com/karpathy/llm.c) using Gaudi2.

## Table Of Contents
* [TPC Kernels Overview](#tpc-kernels-overview)
* [Install Habanatools For Ubuntu](#install-habanatools-for-ubuntu)
* [llm.c Example of Layer Norm](#llmc-examples)
* [Build TPC + Glue code using CMake for Gaudi2](#cmake-build)

## TPC Kernels Overview
The Tensor Processor Core™ (**TPC**) is a fully programmable VLIW4 processor designed to execute non-linear deep learning operators. It is embedded in Habana’s Gaudi deep learning accelerator. Habana’s Gaudi SoC contains numerous TPC cores all operating in parallel, with each core running a single thread. The TPC is designed with very long instruction word (VLIW) architecture. It has a wide single instruction multiple data (SIMD) vector unit that support 2048-bit SIMD operations with data types such as float, bfloat16, INT16, INT32 and INT8. In each cycle, the TPC’s ALU (Arithmetic Logic Unit) can execute up to 64 floats/INT32 ops, or 128 INT16 ops, or 256 INT8 ops.
TPC is designed for workloads that do not map to Matrix Multiplication Engine (**MME**). Those workloads or operators can be implemented using TPC kernels. 

## Install Habanatools For Ubuntu
To retrieve the package please visit [Habana Vault](https://vault.habana.ai/artifactory/debian/jammy/pool/main/h/habanatools/habanatools_1.16.0-526_amd64.deb), click Artifact, find habanatools and download the latest release package for Ubuntu 22.04. You can find different packages for different OS you used. 
```  
  sudo dpkg -i ./habanatools_1.16.0-526_amd64.deb
```
- Once installed the following files will be added to your machine 
  
  |  |Location | Purpose  |
  |--|--------------------|-----------------------------|
  |1 | /usr/bin/tpc-clang | TPC-C compiler and assembler |
  |2 | /usr/bin/tpc-llvm-objdump | TPC dis-assembler|
  |3 | /usr/lib/habanatools/libtpcsim_shared.so | TPC simulator|
  |4 | /usr/lib/habanatools/libtpc_tests_core.so | Test core library |  
  |5 | /usr/lib/habanatools/include/gc_interface.h | Glue code interface header |
  |6 | /usr/lib/habanatools/include/tpc_kernel_lib_interface.h | New TPC kernel GC2.0 interface header |
  |7 | /usr/lib/habanatools/include/tpc_test_core_api.h |Test core APIs |
  |8 | /usr/lib/habanatools/include/tpc_test_core_types.h | Test core type defines |  

### llm.c Example of Layer Norm

- Compiler usage example
The compiler supports a single translation unit, hence ‘-c’ argument should be defined.
```  
 /usr/bin/tpc-clang layernorm_fwd.c -c -x c++ -o layernorm_fwd.o
```  
The output of the compilation session will be an elf file named ‘batch_norm_fwd_f32.o’ . To extract raw binary, from the elf, use the following command:
```  
 objcopy -O binary --only-section=.text layernorm_fwd.o layernorm_fwd.bin 
```  
Using cmake tool shown in the following template examples.
    
For other OS, please refer to the [TPC Tools Installation Guide](https://docs.habana.ai/en/latest/TPC_Tools_Installation/TPC_Tools_Installation_Guide.html) for more details. If you get error like can't find libTpcElfReader.so etc, make sure you add /usr/lib/habanatools path to LD_LIBRARY_PATH environment variable.

The template examples show users how to create and build the custom kernels, which can be used in Tensorflow (**TF**) and PyTorch (**PT**) custom ops later.
This template example has organized in the following way, which contains **TPC kennels**(kernels/), **Glue codes**(src/) and **Unit tests**(tests/).
* TPC kernel codes are the ISA executed by the TPC processor. They contain the kernel implementation.
* Glue codes are executed on the host machine serviced by the Habana DNN SoC, and they hold specifications regarding how the program input/outputs can be dynamically partitioned between the numerous TPC processors in the Habana device.
* Unit tests are to verify the kernel's correctness using the build-in simulator provided in the HabanaTools, test core provides the ability to test on real device and performance.

###Build TPC + Glue code using CMake for Gaudi2

Make sure your Habana tools are installed, check the /usr/bin/tpc-clang and Cmake are up-to-date version, you can download latest cmake via <https://cmake.org/download/>

Clone the repository
```  
 git clone https://github.com/abhilash1910/llm.tpc.git
``` 
In the terminal, make sure you are in the project root directory, then create a directory called build
```  
mkdir build
cd build
```  
then run the following commands
```  
cmake ..
make
```  
After build, you can find libcustom_tpc_perf_lib.so in build/src directory, which is your custom kernel library.
For more details about TPC kernel writing, please refer to the [TPC User Guide](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html) for more information.
