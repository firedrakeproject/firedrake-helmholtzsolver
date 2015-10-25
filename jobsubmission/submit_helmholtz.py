import sys
import os
from optparse import OptionParser

from jobscript import Jobscript

def LogicalStr(x):
    '''Convert locical variable to string (i.e. True -> 'True' and
    False -> 'False')

    :arg x: Variable to be converted
    '''
    if x:
        return 'True'
    return 'False'

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

def weak_scaling(rundir,higher_order,singlelevel=False):
    '''Generate files for a weak scaling experiment.

    :arg rundir: directory to run in
    :arg higher_order: Use higher order discretisation?
    '''
    if (higher_order == True):
        nlevel_list = (2,3,3,3,3,3)
        ref_count_coarse_list = (0,0,1,2,3,4)
        ppn_list = (1,6,24,24,24,24)
        nodes_list = (1,1,1,4,16,64)
    else:
        nlevel_list = (4,4,4,4,4,4,4)
        ref_count_coarse_list = (0,1,2,3,4,5,6)
        ppn_list = (1,6,24,24,24,24,24)
        nodes_list = (1,1,1,4,16,64,256)

    ncoarsesmooth=1
    if (singlelevel):
        ref_count_coarse_list = [x+y for x,y in zip(nlevel_list,
                                                    ref_count_coarse_list)]
        nlevel_list = [0 for x in zip(nlevel_list)]

    nu_cfl = 4.0
    for ref_count_coarse, nlevel, ppn, nodes in zip(ref_count_coarse_list,
                                            nlevel_list,
                                            ppn_list,
                                            nodes_list):
        d = {'ref_count_coarse':ref_count_coarse,
             'higher_order':higher_order,
             'n_level':nlevel,
             'nu_cfl':nu_cfl,
             'ppn':ppn,
             'nodes':nodes,
             'ncoarsesmooth':ncoarsesmooth,
             'multigrid':LogicalStr(not singlelevel)}
        nprocs = ppn*nodes
        label = 'nproc'+str(nprocs)
        create_submission(rundir,label,d)

def vary_cfl(rundir,higher_order,singlelevel=False):
    '''Generate files for runs with different CFL numbers

    :arg rundir: directory to run in
    :arg higher_order: Use higher order discretisation?
    '''
    ppn = 24
    nodes = 1
    if (higher_order == True):
        n_level = 3
        ref_count_coarse = 1
    else:
        n_level = 4
        ref_count_coarse = 2

    for nu_cfl in (2., 4., 6., 8., 16., 32., 64.):
        if (singlelevel):
            ncoarsesmooth=1
        else:
            ncoarsesmooth=int(nu_cfl)/2
        d = {'ref_count_coarse':ref_count_coarse,
             'higher_order':higher_order,
             'n_level':n_level,
             'nu_cfl':nu_cfl,
             'ppn':ppn,
             'nodes':nodes,
             'ncoarsesmooth':ncoarsesmooth,
             'multigrid':LogicalStr(not singlelevel)}
        label = 'CFL'+('%6.3f' % nu_cfl).strip()
        create_submission(rundir,label,d)

#################################################################
# M A I N
#################################################################
if (__name__ == '__main__'):

    # Parse command line and set parameters

    parser = OptionParser('python '+sys.argv[0]+' [options] DIRECTORY: Generate submission scripts and parameter files in directory DIRECTORY')

    parser.add_option('-o','--order', dest='order',
                  type='choice',
                  choices=('lowest','higher'),
                  default='lowest',
                  help='order of finite elements [lowest,higher]')

    parser.add_option('-s', '--singlelevel',
                      action='store_true', dest='singlelevel', default=False,
                      help='use single level method?')

    parser.add_option('-w','--weakscaling',
                      action='store_true', dest='weakscaling', default=False,
                      help='generate files for weak scaling run?')

    parser.add_option('-v','--varycfl',
                      action='store_true', dest='varycfl', default=False,
                      help='generate files for runs with different CFL numbers?')

    (options,args) = parser.parse_args()

    if (len(args) != 1):
        parser.print_help()
        sys.exit(1)

    rundir = args[0]

    higher_order = (options.order == 'higher')

    print 'Run directory = '+rundir
    print 'order = '+options.order
    print 'singlelevel = '+str(options.singlelevel)

    # Create directory if it does not exist
    if not os.path.exists(rundir):
        print 'creating directory \"'+rundir+'\"'
        os.makedirs(rundir)

    if (options.weakscaling):
        print 'Generating files for weak scaling run...'
        weak_scaling(rundir,higher_order,options.singlelevel)
    if (options.varycfl):
        print 'Generating files for run with varying CFL number...'
        vary_cfl(rundir,higher_order,options.singlelevel)
