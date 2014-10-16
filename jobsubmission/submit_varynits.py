import os
import sys
from jobscript import Jobscript

###################################################
# Convert bool to string
###################################################
def asBoolStr(x):
    if x:
        return 'True'
    else:
        return 'False'

###################################################
# Create parameter file 
###################################################
def create_parameterfile(rundir,
                         higher_order,
                         solver,
                         mass_lumping,
                         nits):
    parameterfilename = '' 
    if higher_order:
        parameterfilename += 'higherorder'
    else:
        parameterfilename += 'lowestorder'
    parameterfilename += '_'+solver+'_'
    if (mass_lumping):
        parameterfilename += 'lumped_'
    parameterfilename += str(nits)+'.dat'
    parameterdirectory = os.path.join(rundir,'input')
    parameterfilename = os.path.join(parameterdirectory,parameterfilename)
    try:
        os.stat(parameterdirectory)
    except:
        os.mkdir(parameterdirectory)
    d = {'higher_order':asBoolStr(higher_order),
         'lump_mass':asBoolStr(mass_lumping),
         'maxiter_inner':nits,
         'inner_solver':solver}
    with open('parameters_varynits.tpl') as templatefile:
        template = templatefile.read()
    with open(parameterfilename,'w') as parameterfile:
        parameterfile.write(template % d)
        parameterfile.flush()
    return parameterfilename

###################################################
# Get output file name 
###################################################
def create_outputfilename(higher_order,
                          solver,
                          mass_lumping,
                          nits):
    s = ''
    if higher_order:
        s += 'HigherOrder'
    else:
        s += 'LowestOrder'
    s += '_'+solver
    if (mass_lumping):
        s += 'L'
    return os.path.join(os.path.join('data',s),str(nits)+'.dat')

###################################################
# Create aprun commands
###################################################
def create_apruncommands(rundir,
                         nits_list,
                         solver_list,
                         mass_lumping_list,
                         higher_order_list):
    s = ''
    for higher_order in higher_order_list:
        for solver in solver_list:
            for mass_lumping in mass_lumping_list:
                if not ( (solver == 'richardson') and (not mass_lumping)):
                    for nits in nits_list:
                        parameterfilename = create_parameterfile(rundir,
                                                                 higher_order,
                                                                 solver,
                                                                 mass_lumping,
                                                                 nits)
                        outputfilename = create_outputfilename(higher_order,
                                                               solver,
                                                               mass_lumping,
                                                               nits)
                        if (nits == 1):
                            s += 'mkdir -p '+os.path.dirname(outputfilename)
                            s += '\n\n'
                        s += 'aprun -n 1 -N 1 -S 1 python ${HELMHOLTZSOURCEDIR}/driver.py '
                        s += parameterfilename 
                        s += ' 2>&1  | tee -a '
                        s += outputfilename
                        s += '\n'
                        s += 'echo -n '+outputfilename
                        s += ' finished at | tee -a $LOGFILE\n'
                        s += 'date | tee -a $LOGFILE\n\n'
    return s

#########################################################################
# M A I N
#########################################################################
if (__name__ == '__main__'):
    nits_list = [1,2,3,4,5,10,100]
    solver_list = ['richardson','cg']
    mass_lumping_list = [True,False]
    higher_order_list = [False,True]

    if (len(sys.argv) != 2):
        print 'Usage: python '+sys.argv[0]+' <rundir>'
        sys.exit(0)
    rundir = sys.argv[1]
    try:
        os.stat(rundir)
    except:
        os.mkdir(rundir)
    jobscriptfilename = rundir+'/varynits.pbs'
    runcmd = create_apruncommands(rundir,
                                  nits_list,
                                  solver_list,
                                  mass_lumping_list,
                                  higher_order_list)
    # Create job script
    job = Jobscript('jobscript_varynits.tpl',
                    jobname='varynits',
                    nodes=1,
                    ppn=1,
                    walltime_hours=6,
                    walltime_minutes=0,
                    queue='standard',
                    apruncmd=runcmd)
    job.save_to_file(jobscriptfilename)
