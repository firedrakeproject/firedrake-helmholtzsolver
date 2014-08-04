from mpi4py import MPI

class Logger(object):
    '''Parallel output class.

    This class can be used to selectively print output messages on
    one MPI rank only.
    '''
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def write(self,text):
        '''Print text to screen on master rank only.

        :arg text: text to print to screen
        '''
        if (self.rank == 0):
            print text 
