from mpi_utils import Logger

class Parameters(object):
    '''Class for storing a set of parameters.

    The actual parameters are given as a dictionary

    :arg label: Name of parameter set.
    '''

    def __init__(self,label,data):
        self.label = label
        self._data = data
        self.logger = Logger()
        self.show()

    def show(self):
        '''Print parameters to screen.
        '''
        self.logger.write(' Parameters: '+str(self.label))
        for (key,value) in self._data.items():
            self.logger.write('     '+str(key)+' = '+str(value))
        self.logger.write('')

    def __getitem__(self,key):
        '''Get the value for a specific key.
    
        :arg key: Key to look for
        '''
        return self._data[key]
