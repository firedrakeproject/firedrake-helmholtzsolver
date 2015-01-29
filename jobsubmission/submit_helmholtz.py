import sys

from jobscript import Jobscript

#################################################################
# M A I N
#################################################################
if (__name__ == '__main__'):

    # Parse command line and set parameters
    if (len(sys.argv) != 2):
        print 'Usage: python '+sys.argv[0]+' <directory>'
        sys.exit(1)
    rundir = sys.argv[1]

    # Weak scaling experiment
    ref_count_coarse_list = (4,5,6)
    ppn_list = (6,24,24)
    nodes_list = (1,1,4)

    for ref_count_coarse, ppn, nodes in zip(ref_count_coarse_list,
                                           ppn_list,
                                           nodes_list):
        nprocs = ppn*nodes
        parameterfilename = rundir+'/parameters_nproc'+str(nprocs)+'.in'
        jobscriptfilename = rundir+'/helmholtz_nproc'+str(nprocs)+'.pbs'
        d = {'ref_count_coarse':ref_count_coarse}
        with open('parameters_helmholtz.tpl') as templatefile:
            template = templatefile.read()
        with open(parameterfilename,'w') as parameterfile:
            parameterfile.write(template % d)
            parameterfile.flush()

        # Create job script
        job = Jobscript('jobscript_helmholtz.tpl',
                        jobname='helmholtz',
                        nodes=nodes,
                        ppn=ppn,
                        queue='standard',
                        parameterfilename=parameterfilename)
        job.save_to_file(jobscriptfilename)

