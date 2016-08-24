import sys
'''Utilities for submitting parallel jobs on ARCHER.
'''

class Jobscript(object):
    def __init__(self,
                 templatefilename,
                 jobname='helmholtz',
                 ppn=1,
                 nodes=1,
                 walltime_minutes=5,
                 walltime_hours=0,
                 queue='short',
                 subdirs=('',),
                 parameterfilenames=('parameters.in',),
                 petscparameters='',
                 apruncmd=''):
        '''Class representing a job specificatin.
        
            :arg templatefilename: Name of template file to use
            :arg jobname: Label of job
            :arg ppn: Number of processors per node
            :arg nodes: Number of nodes
            :arg walltime_hours: Walltime (hours)
            :arg walltime_minutes: Walltime (minutes)
            :arg queue: Queue to run in
            :arg subdirs: Names of subdirectories to run in
            :arg parameterfilenames: Names of parameter files
            :arg petscparameters: PETSc parameters to be added to the command line
            :arg apruncmd" aprun command
        '''
        self.apruncmd=apruncmd
        self.jobname=jobname
        self.ppn=ppn
        self.nodes=nodes
        self.subdirs = subdirs
        self.parameterfilenames = parameterfilenames
        self.walltime_minutes = walltime_minutes
        self.walltime_hours = walltime_hours
        self.queue = queue
        self.templatefilename = templatefilename
        self.petscparameters = petscparameters

    def save_to_file(self,filename):
        '''Save jobscript to disk.
            
            :arg filename: Name of file to write.
        '''
        with open(self.templatefilename) as templatefile:
            template = templatefile.read()
        s = ''
        for subdir, parameterfile in zip(self.subdirs,self.parameterfilenames):
            s += 'WORKSUBDIR=$WORKDIR/'+subdir+'\n'
            s += 'PARAMETERFILE='+parameterfile+'\n'
            s += 'PETSCPARAMETERS='+self.petscparameters+'\n'
            s += 'mkdir $WORKSUBDIR\n'
            s += 'cp $0 $WORKSUBDIR/jobscript.pbs\n'
            s += 'cp $PBS_O_WORKDIR/$PARAMETERFILE $WORKSUBDIR\n'
            s += 'cd $WORKSUBDIR\n'
            s += 'aprun -n '+str(self.ppn*self.nodes)+' -N '+str(self.ppn)+' -S '+str((self.ppn+1)/2)+' python ${HELMHOLTZSOURCEDIR}/driver.py $PARAMETERFILE $PETSCPARAMETERS 2>&1  | tee -a output.log\n\n'
        d = {'queue':self.queue,
             'nodes':self.nodes,
             'ppn':self.ppn,
             'ptotal':self.ppn*self.nodes,
             'pnuma':(self.ppn+1)/2,
             'jobname':self.jobname,
             'walltime_hours':self.walltime_hours,
             'walltime_minutes':self.walltime_minutes,
             'apruncmd':self.apruncmd,
             'subruns':s}
        
        with open(filename,'w') as jobfile:
            jobfile.write(template % d)
            jobfile.flush()
