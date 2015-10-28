#!/bin/bash --login

#PBS -N %(jobname)s
#PBS -l walltime=%(walltime_hours)d:%(walltime_minutes)d:0
#PBS -l select=%(nodes)d
#PBS -A n02-NEJ005576
#PBS -M e.mueller@bath.ac.uk
#PBS -q %(queue)s

# Make sure any symbolic links are resolved to absolute path 
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Name of job
JOBNAME=%(jobname)s
WORKDIR=$PBS_O_WORKDIR/%(jobname)s_${PBS_JOBID}
cd $PBS_O_WORKDIR

# Directory containing the python sources of the Helmholtz solver module
HELMHOLTZSOURCEDIR=${WORK}/git_workspace/firedrake-helmholtzsolver/source

mkdir -p $WORKDIR
LOGFILE=$WORKDIR/output.log

module use /home/n02/n02/eike/modules
module load firedrake-local

export CC=cc
export CXX=CC
export FIREDRAKE_FFC_KERNEL_CACHE_DIR=$WORK/firedrake-cache
export PYOP2_DEBUG=0
export PYOP2_NO_FORK_AVAILABLE=1
export PYOP2_LAZY=0
export PYOP2_BACKEND_COMPILER=gnu
export PYOP2_SIMD_ISA=avx
export PYOP2_CACHE_DIR=$WORK/pyop2-cache
export LD_LIBRARY_PATH=$ANACONDA_LIB:$LD_LIBRARY_PATH
export PYTHONPATH=$HELMHOLTZSOURCEDIR:${PYTHONPATH}
export PETSC_OPTIONS=-log_summary
# Prevent matplotlib from accessing /home
export HOME=$WORK
export XDG_CONFIG_HOME=''

# MPI (man intro_mpi)
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_MAX_THREAD_SAFETY=multiple
unset MPICH_CPUMASK_DISPLAY

echo | tee -a $LOGFILE
echo Running helmholtz 2>&1  | tee -a $LOGFILE
echo | tee -a $LOGFILE
echo "PBS_JOBID = ${PBS_JOBID}" 2>&1  | tee -a $LOGFILE
echo | tee -a $LOGFILE

echo -n Started at | tee -a $LOGFILE
date | tee -a $LOGFILE

%(subruns)s

echo -n Finished at | tee -a $LOGFILE
date | tee -a $LOGFILE

