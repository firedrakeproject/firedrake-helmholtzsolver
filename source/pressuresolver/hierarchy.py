import xml.etree.cElementTree as ET

class HierarchyContainer(object):
    '''Container class for hierarchy over levels.

    Provides a templated container which can be used to create and store a set
    of objects of type :class:`Type`, each of which is based on one entry in 
    a provided list of different base objects (e.g. functions spaces and/
    or operators on the different levels of a multigrid hierarchy) and 
    a set of common parameters.

    In the constructor, loop over the list of base objects and for each
    element of this list create a new object of type :class:`Type` from both 
    the base object and the common parameters.

    :arg Type: type of objects to be created
    :arg h_args: zipped list of objects to create 
    :arg args: common parameters used in the construction of Type objects
    :arg kwargs: common keyword parameters used in the construction of
        Type objects.
    '''
    def __init__(self,Type,h_args,*args,**kwargs):
        # Construct list of parameters to be used in the
        # individual Type constructors
        arglist = []
        for x in h_args:
            tmp = list(x)
            tmp.extend(args)
            arglist.append(tmp)
        self._data = [Type(*x,**kwargs) for x in arglist] 

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        for i,x in enumerate(self._data):
            x.add_to_xml(e,'level_'+('%03d' % i))
       
    def __getitem__(self,level):
        '''Return object on given level in the functionspace hierarchy.

        :arg level: level in hierarchy
        '''
        return self._data[level]

    def __len__(self):
        '''Return number of levels in operator hierarchy.'''
        return len(self._data)
        
