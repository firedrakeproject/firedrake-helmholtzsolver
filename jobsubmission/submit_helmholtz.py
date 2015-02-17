import sys

from jobscript import Jobscript

def create_submission(rundir,label,d):
    '''Generate submission files (jobscript and parameter file)
    for a given parameter set.

    :arg rundir: directory to run in
    :arg label: Unique label for run
    :arg d: Dictionary with parameters
    '''
    ppn = d['ppn']
    nodes = d['nodes']
    higher_order=d['higher_order']
    nprocs = ppn*nodes
    parameterfilename = rundir+'/parameters_'+label+'.in'
    jobscriptfilename = rundir+'/helmholtz_'+label+'.pbs'
    with open('parameters_helmholtz.tpl') as templatefile:
        template = templatefile.read()
    with open(parameterfilename,'w') as parameterfile:
        parameterfile.write(template % d)
        parameterfile.flush()

    # Create job script
    job = Jobscript('jobscript_helmholtz.tpl',
                    walltime_minutes=15,
                    jobname='helmholtz',
                    nodes=nodes,
                    ppn=ppn,
                    queue='standard',
                    parameterfilename=parameterfilename)
    job.save_to_file(jobscriptfilename)

def weak_scaling(rundir,higher_order):
    '''Generate files for a weak scaling experiment.

    :arg rundir: directory to run in
    :arg higher_order: Use higher order discretisation?
    '''
    if (higher_order == True):
        n_level = 3
        ref_count_coarse_list = (0,1,2,3,4)
        ppn_list = (6,24,24,24,24)
        nodes_list = (1,1,4,16,64)
        pressure_ksp = 'cg'
        pressure_maxiter = 3
    else:
        n_level = 4
        ref_count_coarse_list = (0,1,2,3,4,5,6)
        ppn_list = (1,6,24,24,24,24,24)
        nodes_list = (1,1,1,4,16,64,256)
        pressure_ksp = 'preonly'
        pressure_maxiter = 1

    nu_cfl = 2.0
    for ref_count_coarse, ppn, nodes in zip(ref_count_coarse_list,
                                            ppn_list,
                                            nodes_list):
        d = {'ref_count_coarse':ref_count_coarse,
             'higher_order':higher_order,
             'n_level':n_level,
             'nu_cfl':nu_cfl,
             'pressure_ksp':pressure_ksp,
             'pressure_maxiter':pressure_maxiter,
             'ppn':ppn,
             'nodes':nodes}
        nprocs = ppn*nodes        
        label = 'nproc'+str(nprocs)
        create_submission(rundir,label,d)

def vary_cfl(rundir,higher_order):
    '''Generate files for runs with different CFL numbers

    :arg rundir: directory to run in
    :arg higher_order: Use higher order discretisation?
    '''
    ppn = 24
    nodes = 1
    if (higher_order == True):
        n_level = 3
        ref_count_coarse = 1
        pressure_ksp = 'cg'
        pressure_maxiter = 3
    else:
        n_level = 4
        ref_count_coarse = 2
        pressure_ksp = 'preonly'
        pressure_maxiter = 1

    for nu_cfl in (2., 4., 6., 8., 16., 32., 64.):
        d = {'ref_count_coarse':ref_count_coarse,
             'higher_order':higher_order,
             'n_level':n_level,
             'nu_cfl':nu_cfl,
             'pressure_ksp':pressure_ksp,
             'pressure_maxiter':pressure_maxiter,
             'ppn':ppn,
             'nodes':nodes}
        label = 'CFL'+('%6.3f' % nu_cfl).strip()
        create_submission(rundir,label,d)

#################################################################
# M A I N
#################################################################
if (__name__ == '__main__'):

    # Parse command line and set parameters
    if (len(sys.argv) != 2):
        print 'Usage: python '+sys.argv[0]+' <directory>'
        sys.exit(1)
    rundir = sys.argv[1]

    higher_order = False

#    weak_scaling(rundir,higher_order)
    vary_cfl(rundir,higher_order)
