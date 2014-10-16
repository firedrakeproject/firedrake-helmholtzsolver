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

# Directory containing the python sources of the Helmholtz solver module
HELMHOLTZSOURCEDIR=${WORK}/git_workspace/firedrake-helmholtzsolver/source

cd $PBS_O_WORKDIR

WORKDIR=$PBS_O_WORKDIR/%(jobname)s_${PBS_JOBID}
mkdir -p $WORKDIR

cp $0 $WORKDIR/jobscript.pbs
cp %(parameterfile)s $WORKDIR

cd $WORKDIR

LOGFILE=output.log

module swap PrgEnv-cray PrgEnv-gnu
module unload python
module add anaconda
module use /work/y07/y07/fdrake/modules
module load firedrake

module load fdrake-build-env
module load fdrake-python-env

export PETSC_DIR=/work/n02/n02/eike/git_workspace/petsc
export PETSC_ARCH=cray-gnu-shared
export FIREDRAKE_FFC_KERNEL_CACHE_DIR=$WORK/firedrake-cache
export PYOP2_DEBUG=1
export PYOP2_LAZY=0
export PYOP2_BACKEND_COMPILER=gnu
export PYOP2_SIMD_ISA=avx
export PYOP2_CACHE_DIR=$WORK/pyop2-cache
export LD_LIBRARY_PATH=$ANACONDA_LIB:$LD_LIBRARY_PATH
export FDRAKEWORK=${WORK}/git_workspace/
export PYTHONPATH=$FDRAKEWORK/firedrake-bench:${PYTHONPATH}
export PYTHONPATH=$FDRAKEWORK/pybench:${PYTHONPATH}
export PYTHONPATH=${WORK}/Library/mpi4py/lib/python2.7/site-packages/:${PYTHONPATH}
export PYTHONPATH=$FDRAKEWORK/petsc4py/cray-gnu-shared/lib/python2.7/site-packages/:${PYTHONPATH}
export PYTHONPATH=$FDRAKEWORK/PyOP2:${PYTHONPATH}
export PYTHONPATH=$FDRAKEWORK/firedrake:${PYTHONPATH}
export PYTHONPATH=$HELMHOLTZSOURCEDIR:${PYTHONPATH}
export PETSC_OPTIONS=-log_summary
# Prevent matplotlib from accessing /home
export HOME=$WORK
export XDG_CONFIG_HOME=''

# MPI (man intro_mpi)
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_CPUMASK_DISPLAY=1

echo -n Started at | tee -a $LOGFILE
date | tee -a $LOGFILE

echo | tee -a $LOGFILE
echo Running helmholtz 2>&1  | tee -a $LOGFILE
echo | tee -a $LOGFILE
echo "PBS_JOBID = ${PBS_JOBID}" 2>&1  | tee -a $LOGFILE
echo | tee -a $LOGFILE

aprun -n %(ptotal)d -N %(ppn)d -S %(pnuma)d python ${HELMHOLTZSOURCEDIR}/driver.py %(parameterfile)s 2>&1  | tee -a $LOGFILE

echo -n Finished at | tee -a $LOGFILE
date | tee -a $LOGFILE

