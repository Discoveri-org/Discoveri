export OMPI_CXX=clang++
export HDF5_ROOT=/Users/francescomassimo/Codes/hdf5-1.10.9/build
export HDF5_ROOT_DIR=/Users/francescomassimo/Codes/hdf5-1.10.9/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/francescomassimo/Codes/hdf5-1.10.9/build/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/homebrew/opt/python@3.10/bin/python3.11/lib


export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

export LD_LIBRARY_PATH="/opt/homebrew/opt/llvm/lib":$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH="/opt/homebrew/opt/llvm/include":$CPLUS_INCLUDE_PATH

export OMP_NUM_THREADS=1

nohup mpirun -n 1 smilei namelist_optimization_LWFA.py > smilei.log 2>&1 &

