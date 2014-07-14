import numpy as np 

'''Code generator for a C-kernel which inverts a 3x3 matrix in-place.
'''

class MatrixEntry(object):
    '''A matrix entry.

    Represent the entry of a matrix with indices (i,j).
    '''
    def __init__(self,i,j,label):
        self.i = i
        self.j = j
        self.label = label
    
    def __str__(self):
        '''Convert to string
        '''
        return self.label+'[3*'+str(self.i)+'+'+str(self.j)+']'

def Minor(a,i,j):
    '''Generate string expression for i,j minor if matrix a.
    '''
    sign = (-1)**(i+j)
    b = np.delete(a,i,axis=0)
    b = np.delete(b,j,axis=1)
    s = '('+str(b[0][0])+'*'+str(b[1][1])+'-'+str(b[0][1])+'*'+str(b[1][0])+')'
    if sign == 1:
        s = '+'+s
    else:
        s = '-'+s
    return s

def Determinant(a):
    '''Generate string expression for determinant of matrix a.
    '''
    s = ''
    for i in range(3):
        s += '+'+str(a[0][i])+'*('+Minor(a,0,i)+')'
    return s

def GenerateMat3x3InverseKernel(a):
    '''Generate kernel code for inverting matrix a.
    '''
    s = ''
    s += '// Calculate inverse of 3x3 matrix stored in \n'
    s += '// row-major order, i.e. a_{i,j} = a[3*i+j]\n'
    s += '{\n'
    s += '  double b[9];\n'
    s += '  for (int i=0;i<9;++i) b[i] = a[i];\n'
    for i in range(3):
        for j in range(3): 
            s += '  '+str(MatrixEntry(i,j,'a'))+' = '+Minor(a,j,i)+';\n'
    s += '  double invDet = 1./('+Determinant(a)+')'+';\n'
    s += '  for (int i=0;i<9;++i) a[i] *= invDet;\n'
    s += '}\n'
    return s

# Create symbolic matrix a
a = np.empty(shape=(3,3),dtype=MatrixEntry)
for i in range(3):
    for j in range(3):
        a[i][j] = MatrixEntry(i,j,'b')

# Generate C-kernel code for inverting this matrix in-place
print GenerateMat3x3InverseKernel(a)
