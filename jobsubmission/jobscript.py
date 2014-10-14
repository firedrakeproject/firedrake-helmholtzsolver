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
                 queue='debug',
                 parameterfilename='parameters.in',
                 apruncmd=''):
        '''Class representing a job specificatin.
        
            :arg templatefilename: Name of template file to use
            :arg jobname: Label of job
            :arg ppn: Number of processors per node
            :arg nodes: Number of nodes
            :arg walltime_hours: Walltime (hours)
            :arg walltime_minutes: Walltime (minutes)
            :arg queue: Queue to run in
            :arg parameterfilename: Name of parameter file
            :arg apruncmd" aprun command
        '''
        self.apruncmd=apruncmd
        self.jobname=jobname
        self.ppn=ppn
        self.nodes=nodes
        self.parameterfilename = parameterfilename
        self.walltime_minutes = walltime_minutes
        self.walltime_hours = walltime_hours
        self.queue = queue
        self.templatefilename = templatefilename

    def save_to_file(self,filename):
        '''Save jobscript to disk.
            
            :arg filename: Name of file to write.
        '''
        with open(self.templatefilename) as templatefile:
            template = templatefile.read()
        d = {'queue':self.queue,
             'nodes':self.nodes,
             'ppn':self.ppn,
             'ptotal':self.ppn*self.nodes,
             'pnuma':(self.ppn+1)/2,
             'jobname':self.jobname,
             'parameterfile':self.parameterfilename,
             'walltime_hours':self.walltime_hours,
             'walltime_minutes':self.walltime_minutes,
             'apruncmd':self.apruncmd}
        with open(filename,'w') as jobfile:
            jobfile.write(template % d)
            jobfile.flush()
