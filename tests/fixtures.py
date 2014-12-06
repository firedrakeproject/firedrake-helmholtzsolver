from firedrake import *
import pytest


@pytest.fixture
def mesh():
    '''Create 1+1 dimensional mesh by extruding a circle.'''
    D = 0.1
    nlayers=4
    ncells=3

    host_mesh = CircleManifoldMesh(ncells)
    mesh = ExtrudedMesh(host_mesh,
                        layers=nlayers,
                        extrusion_type='radial',
                        layer_height=D/nlayers)

    return mesh

@pytest.fixture
def finite_elements():
    '''Create finite elements of horizontal and vertical function spaces.

    :math:`U_1` = horizontal H1 space
    :math:`U_2` = horizontal L2 space
    :math:`V_0` = vertical H1 space
    :math:`V_1` = vertical L2 space
    '''
    # Finite elements

    # Horizontal elements
    U1 = FiniteElement('CG',interval,2)
    U2 = FiniteElement('DG',interval,1)

    # Vertical elements
    V0 = FiniteElement('CG',interval,2)
    V1 = FiniteElement('DG',interval,1)

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

