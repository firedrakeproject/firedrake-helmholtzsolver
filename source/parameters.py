from pressuresolver.mpi_utils import Logger
from mpi4py import MPI
import re

class Parameters(object):
    '''Class for storing a set of parameters.

    The actual parameters are given as a dictionary

    :arg label: Name of parameter set.
    '''

    def __init__(self,label,data):
        self.label = label
        self._data = data
        self.logger = Logger()

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

    def read_from_file(self,filename):
        '''Read parameters from file.

        The parameters of the class with a specific label
        are stored as follows (the indent is mandatory)::
        
            `code'
            <label>:
                parameter_1 = value_1
                parameter_2 = value_2
                [...]
                parameter_n = value_n

        :arg filename: Name of file to read
        '''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Read file on rank 0
        if (rank == 0):
            param_file = open(filename,'r')
            # Parse until the keyword <label>: is found
            parse_section = False
            nsection = 0
            valid_keys = {}
            for key in self._data:
                valid_keys[key] = False
            for line in param_file:
                # Find section labels in parameter file
                m = re.match('^ *([a-zA-Z0-9\_\- ]+): *$',line)
                if m:
                    parse_section = (str(m.group(1)) == self.label)
                    if (parse_section):
                        nsection += 1
                # Read data
                if parse_section:
                    m = re.match(' +([a-zA-Z0-9\_]+) *= *([0-9a-zA-Z\_\-\+\.]+) *[#]?.*',line)
                    if m:
                        key, value = m.group(1), m.group(2)
                        if (key in self._data.keys()):
                            value = self._parse_value(value)
                            valid_type = ((type(value) == type(self._data[key])) or \
                                          ((type(value) is int) and (type(self._data[key]) is float)))
                            if valid_type:
                                self._data[key] = value 
                                valid_keys[key] = True
                            else:
                                print type(value), type(self._data[key])
            # Section was not found or was found multiple times
            if (nsection==0):
                self.logger.warning('parameters for '+self.label+\
                                    ' not defined in file '+filename+'.')
            if (nsection>1):
                self.logger.warning('parameters for '+self.label+\
                                    ' defined multiple times in '+filename+'.')
            # Check if all parameters in the section were found
            for key, valid in valid_keys.iteritems():
                if (not valid):
                    self.logger.warning('No valid value provided for parameter '+\
                                        self.label+'::'+key+' in file'+filename+'.')
                
            param_file.close()
        self._data = comm.bcast(self._data,root=0)

    def _parse_value(self,value):
        '''Convert a parameter value provided as a string to valid python data type.

        If it is not either integer, float or bool, simply return the input data string
    
        :arg value: string to parse
        '''
        # Bool
        if (value == 'True'):
            return True
        if (value == 'False'):
            return False
        # Try integer
        m = re.match('^[\-\+]?[0-9]+$',value)
        if m:
            return int(value)
        try:
            return float(value)
        except:
            return value
        return value
