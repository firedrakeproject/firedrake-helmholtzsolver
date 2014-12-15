from firedrake import *
from ffc import log
log.set_level(log.ERROR)
op2.init(log_level="WARNING")
import pytest

@pytest.fixture(params=[2,3])
def host_mesh(request):
    '''Create host mesh, which is either a circle or a sphere.'''

    dimension = request.param

    if (dimension == 2):
        ncells=3
        host_mesh = CircleManifoldMesh(ncells)
    else:
        refcount = 0
        host_mesh = UnitIcosahedralSphereMesh(refcount)
    return host_mesh

@pytest.fixture
def mesh(host_mesh):
    '''Create extruded mesh.

    :arg host_mesh: Mesh to be extruded
    '''
    D = 0.1
    nlayers = 4
    mesh = ExtrudedMesh(host_mesh,
                        layers=nlayers,
                        extrusion_type='radial',
                        layer_height=D/nlayers)

    return mesh

@pytest.fixture
def mesh_hierarchy(host_mesh):
    '''Create mesh hierarchy'''
    D = 0.1
    nlayers = 4
    nlevel = 4
    dimension = host_mesh._ufl_cell.topological_dimension()+1
    if (dimension == 3):
        host_mesh_hierarchy = MeshHierarchy(host_mesh,nlevel)
        mesh_hierarchy = ExtrudedMeshHierarchy(host_mesh_hierarchy,
                                               layers=nlayers,
                                               extrusion_type='radial',
                                               layer_height=D/nlayers)
    else:
        mesh_hierarchy = None

    return mesh_hierarchy

@pytest.fixture
def pressure_expression(mesh):
    '''Analytical expression for a pressure function

    :arg mesh: underlying extruded mesh (needed for extracting dimension)
    '''
    dimension = mesh._ufl_cell.topological_dimension()
    if (dimension == 2):
        return Expression('x[0]+2.0*x[1]*x[1]')
    else:
        return Expression('x[0]+2.0*x[1]*x[1]+0.5*x[2]*x[0]')

@pytest.fixture
def velocity_expression(mesh):
    '''Analytical expression for a velocity function

    :arg mesh: underlying extruded mesh (needed for extracting dimension)
    '''
    dimension = mesh._ufl_cell.topological_dimension()
    if (dimension == 2):
        return Expression(('x[0]+2.0*x[1]','(x[0]+1.0)*x[1]'))
    else:
        return Expression(('x[0]+2.0*x[1]+3.0*x[2]','(x[0]+1.0)*x[1]*x[2]','x[2]*x[0]+x[1]'))

@pytest.fixture
def finite_elements(mesh):
    '''Create finite elements of horizontal and vertical function spaces.

    :math:`U_1` = horizontal H1 space
    :math:`U_2` = horizontal L2 space
    :math:`V_0` = vertical H1 space
    :math:`V_1` = vertical L2 space

    :arg mesh: underlying extruded mesh (needed for extracting dimension)
    '''
    dimension = mesh._ufl_cell.topological_dimension()
    # Finite elements
    # Horizontal elements
    if (dimension == 2):
        U1 = FiniteElement('CG',interval,2)
        U2 = FiniteElement('DG',interval,1)
    else:
        U1 = FiniteElement('RT',triangle,1)
        U2 = FiniteElement('DG',triangle,0)
        
    # Vertical elements
    if (dimension == 2):
        V0 = FiniteElement('CG',interval,2)
        V1 = FiniteElement('DG',interval,1)
    else:
        V0 = FiniteElement('CG',interval,1)
        V1 = FiniteElement('DG',interval,0)

    return U1, U2, V0, V1

@pytest.fixture
def W2(finite_elements,mesh):
    '''HDiv space for velocity.
            
    Build full velocity space :math:`W_2 = Hdiv(U_1\otimes V_1)\oplus Hdiv(U_2\otimes V_0)`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U1,V1)) + HDiv(OuterProductElement(U2,V0))

    W2 = FunctionSpace(mesh,W2_elt)
    
    return W2

@pytest.fixture
def W2_horiz(finite_elements,mesh):
    '''HDiv space for horizontal velocity component.
            
    Build vertical horizontal space :math:`W_2^{h} = Hdiv(U_1\otimes V_1)`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U1,V1))

    W2_horiz = FunctionSpace(mesh,W2_elt)
    
    return W2_horiz

@pytest.fixture
def W2_vert(finite_elements,mesh):
    '''HDiv space for vertical velocity component.
            
    Build vertical horizontal space :math:`W_2^{v} = Hdiv(U_2\otimes V_0)`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U2,V0))

    W2_vert = FunctionSpace(mesh,W2_elt)
    
    return W2_vert

@pytest.fixture
def Wb(finite_elements,mesh):
    '''Finite element space for buoyancy.
            
    Build vertical horizontal space :math:`W_b = U_2\otimes V_0`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = OuterProductElement(U2,V0)

    Wb = FunctionSpace(mesh,W2_elt)
    
    return Wb


@pytest.fixture
def W3(finite_elements,mesh):
    '''L2 pressure space.
            
    Build pressure space :math:`W_3 = Hdiv(U_2\otimes V_1)`

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W3_elt = OuterProductElement(U2,V1)

    W3 = FunctionSpace(mesh,W3_elt)
    return W3

@pytest.fixture
def W3_hierarchy(finite_elements,mesh_hierarchy):
    '''L2 pressure space hierarchy.
            
    Build pressure space :math:`W_3 = Hdiv(U_2\otimes V_1)` hierarchy

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W3_elt = OuterProductElement(U2,V1)

    if (mesh_hierarchy != None):
        W3_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W3_elt)
    else:
        W3_hierarchy = None
    return W3_hierarchy

@pytest.fixture
def W2_horiz_hierarchy(finite_elements,mesh_hierarchy):
    '''Horizontal velocity space hierarchy.
            
    Build pressure space :math:`W_2^{h}=HDiv(U_1\otimes V_1)` 
    hierarchy.

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U1,V1))

    if (mesh_hierarchy != None):
        W2_horiz_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt)
    else:
        W2_horiz_hierarchy = None
    return W2_horiz_hierarchy

@pytest.fixture
def W2_vert_hierarchy(finite_elements,mesh_hierarchy):
    '''Vertical velocity space hierarchy.
            
    Build pressure space :math:`W_2^{v}=HDiv(U_2\otimes V_0)`
    hierarchy.

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U2,V0))

    if (mesh_hierarchy != None):
        W2_vert_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt)
    else:
        W2_vert_hierarchy = None
    return W2_vert_hierarchy

@pytest.fixture
def W2_hierarchy(finite_elements,mesh_hierarchy):
    '''Hdiv velocity space hierarchy.
            
    Build pressure space :math:`W_2 = HDiv(U_2\otimes V_1)\oplus HDiv(U_1\otimes V_1)` 
    hierarchy.

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U1,V1)) + HDiv(OuterProductElement(U2,V0))

    if (mesh_hierarchy != None):
        W2_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt)
    else:
        W2_hierarchy = None
    return W2_hierarchy

@pytest.fixture
def Wb_hierarchy(finite_elements,mesh_hierarchy):
    '''buoyancy space hierarchy.
            
    Build buoyance space :math:`W_2 = U_2\otimes V_1` hierarchy.

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    Wb_elt = OuterProductElement(U2,V0)

    if (mesh_hierarchy != None):
        Wb_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,Wb_elt)
    else:
        Wb_hierarchy = None
    return Wb_hierarchy

