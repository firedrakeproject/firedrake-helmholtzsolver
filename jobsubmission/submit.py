import sys

from jobscript import Jobscript
from helmholtz_parameters import HelmholtzParameters


#################################################################
# M A I N
#################################################################
if (__name__ == '__main__'):

    # Parse command line and set parameters
    if (len(sys.argv) != 2):
        print 'Usage: python '+sys.argv[0]+' <directory>'
        sys.exit(1)
    rundir = sys.argv[1]

    parameterfilename = rundir+'/parameters.in'
    jobscriptfilename = rundir+'/helmholtz.pbs'

    param = HelmholtzParameters()
    with open(parameterfilename,'w') as parameterfile:
        print >> parameterfile, str(param)

    # Create job script
    job = Jobscript(jobname='helmholtz',
                    ppn=4,
                    parameterfilename=parameterfilename)
    job.save_to_file(jobscriptfilename)

