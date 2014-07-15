import sys
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from firedrake import * 
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers, solvers, preconditioners, lumpedmass_bdfm1

##########################################################
# M A I N
##########################################################
if (__name__ == '__main__'):
    # Parameters
    ref_count = 1
    mu_relax = 1.0
    outputDir = 'output'
    ignore_mass_lumping = False
    higher_order = True  
    
    # Create mesh
    mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count)
    global_normal = Expression(("x[0]","x[1]","x[2]"))
    mesh.init_cell_orientations(global_normal)

    ncells = mesh.num_cells()
    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))

    omega = 8.*0.5*dx
    if (higher_order):
        V_pressure = FunctionSpace(mesh,'DG',1)
        V_velocity = FunctionSpace(mesh,'BDFM',2)
    else:
        V_pressure = FunctionSpace(mesh,'DG',0)
        V_velocity = FunctionSpace(mesh,'RT',1)

    operator = pressuresolver.operators.Operator(V_pressure,V_velocity,
                                                 omega)
    if (higher_order):
        smoother = pressuresolver.smoothers.Jacobi_HigherOrder(operator,
                                                               mu_relax=mu_relax)
    else:
        smoother = pressuresolver.smoothers.Jacobi_LowestOrder(operator,
                                                               mu_relax=mu_relax)

    # Calculate eigenvalues
    b = Function(V_pressure)
    b.assign(0.0)
    f = Function(V_pressure)
    g = Function(V_pressure)
    ndofs = V_pressure.dof_count
    m_eigen = np.zeros((ndofs,ndofs))
    m_smooth = np.zeros((ndofs,ndofs))
    for i in range(ncells):
        f.dat.data[:] = 0 
        f.dat.data[i] = 1
        g = operator.apply(f)
        m_eigen[i,:] = g.dat.data[:]
        smoother.smooth(b,f,initial_phi_is_zero=False)
        m_smooth[i,:] = f.dat.data[:]
    evals = np.linalg.eigvalsh(m_eigen)
    D_diag = 1./smoother.D_diag_inv.dat.data
    max_D_diag = np.max(D_diag)
    min_D_diag = np.min(D_diag)
    max_eval = np.max(evals)
    min_eval = np.min(evals)
    print 'Eigenvalues: {0: 10.4e} ... {1: 10.4e}'.format(min_eval,max_eval)
    print 'Diagonalvalues: {0: 10.4e} ... {1: 10.4e}'.format(min_D_diag,max_D_diag)
    print 'ratio : {0: 10.4}'.format(max_eval/max_D_diag)
    y = np.zeros(ndofs)
    print len(evals), len(y)
    print D_diag
    plt.clf()
    ax = plt.gca()
    ax.set_ylim(-0.3,0.3)
    ax.set_xlim(0,1.1*max_D_diag)
    p1 = plt.plot(evals,y-0.2,color='red',markeredgecolor='red',linewidth=0,markersize=6,marker='o')[0]
    #p2 = plt.plot(D_diag,y+0.2,color='blue',markeredgecolor='blue',linewidth=0,markersize=6,marker='o')[0]
    #plt.legend((p1,p2),('Eigenvalues','Diagonal entries'),'upper left')
    plt.savefig(outputDir+'/eigenvalues.pdf',bbox_inches='tight')

    evals_smoother = np.linalg.eigvals(m_smooth)
    w,v = np.linalg.eig(m_smooth)
    print evals_smoother
    max_smoother = np.max(evals_smoother)
    min_smoother = np.min(evals_smoother)
    print 'Eigenvalues: {0: 10.4e} ... {1: 10.4e}'.format(min_smoother,max_smoother)
    plt.clf()
    ax = plt.gca()
    ax.set_ylim(-0.2,0.2)
    ax.set_xlim(-1.1,1.1)
    p = plt.plot(evals_smoother,y,color='red',markeredgecolor='red',linewidth=0,markersize=6,marker='o')[0]
    plt.plot([-1,-1],[-0.2,0.2],color='black',linewidth=2)
    plt.plot([+1,+1],[-0.2,0.2],color='black',linewidth=2)
    plt.legend((p,),('Eigenvalues smoother',),'upper left')
    plt.savefig(outputDir+'/eigenvalues_smoother.pdf',bbox_inches='tight')
    for i in range(ndofs):
        if (w[i] >= 1.0):
            print w[i],' : ',v[i]

