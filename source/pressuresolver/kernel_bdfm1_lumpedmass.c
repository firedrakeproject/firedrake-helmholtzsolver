void set_matrix(double **m,
                double **coords,
                double **U_x, 
                double **U_y,
                double **U_z,
                double **U_tilde_x, 
                double **U_tilde_y,
                double **U_tilde_z) {
    /* Calculate 4x4 matrix whose row vectors
     * are given by the dofs of four solid body rotation
     * fields, constructed from the flow fields 
     * U_x, U_y, U_z and U_tilde_x, U_tilde_y, U_tilde_z
     * (these could also represent the flow fields
     * multiplied by the mass matrix).
     *
     * INPUT:
     *   - U_x       = (    0,   -z,    y)  }
     *   - U_y       = (    z,    0,   -x)  }  Flow type A
     *   - U_z       = (   -y,    x,    0)  } 
     *   - U_tilde_x = (    0, -z*x,  y*x)  }
     *   - U_tilde_y = (  z*y,    0, -x*y)  }  Flow type B
     *   - U_tilde_z = ( -y*z,  x*z,    0)  } 
     *   - coords: Mesh coordinate field
     * OUTPUT:
     *   - m: 4x4 matrix stored on facets
     */

    /* Step 1: Work out unit normal and tangential vectors
     *  r = (x_0+x_1)/|x_0+x_1| 
     *  t = (x_0-x_1)/|x_0-x_1| ( unit tangential )
     *  n = a x t               ( unit normal)
     */
    double r[3], n[3], t[3];
    double C_r=0.0;
    double C_t=0.0;
    for (int i=0; i<3; ++i) {
        r[i] = coords[0][i] + coords[1][i];
        t[i] = coords[0][i] - coords[1][i];
        C_t += t[i]*t[i];
        C_r += r[i]*r[i];
    } 
    C_t = 1.0/sqrt(C_t);
    C_r = 1.0/sqrt(C_r);
    for (int i=0; i<3; ++i) {
        r[i] *= C_r;
        t[i] *= C_t;
    }
    n[0] = r[1]*t[2] - r[2]*t[1];
    n[1] = r[2]*t[0] - r[0]*t[2];
    n[2] = r[0]*t[1] - r[1]*t[0];

    /* Step 2: Construct solid body rotation fields 
     * perpendicular/tangential to current edge and 
     * store their edge dofs in the rows of
     * the matrix m
     *
     *  (0): t.U       (maximal tangential flow of type A)
     *  (1): n.U       (maximal normal flow of type A)
     *  (2): t.U_tilde (maximal tangential flow of type B)
     *  (3): n.U_tilde (maximal normal flow of type B)
     */
    for (int i=0; i<4; ++i) {
        m[0][0+i] = t[0]*U_x[i][0] 
                  + t[1]*U_y[i][0]
                  + t[2]*U_z[i][0];
        m[0][4+i] = t[0]*U_tilde_x[i][0]
                  + t[1]*U_tilde_y[i][0] 
                  + t[2]*U_tilde_z[i][0];
        m[0][8+i] = n[0]*U_x[i][0] 
                  + n[1]*U_y[i][0]
                  + n[2]*U_z[i][0];
        m[0][12+i] = n[0]*U_tilde_x[i][0] 
                   + n[1]*U_tilde_y[i][0]
                   + n[2]*U_tilde_z[i][0];
    }
}

