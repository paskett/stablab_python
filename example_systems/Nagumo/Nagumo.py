import numpy as np
import matplotlib.pyplot as plt
import time
import stablab

def Avec(x, lamda, s, p):

    u_hat = np.sqrt(2)*1/np.cosh(x)

    on = np.ones(1,dtype=np.complex)
    out = np.zeros((4,4),dtype=np.complex)
    out[0,1] = on
    out[1,0] = lamda+1-3*u_hat**2
    out[1,2] = -p.a*on
    out[2,3] = on
    out[3,0] = -p.b*on
    out[3,2] = lamda+1-3*u_hat**2
    return out

def A_pm(x,lamda,s,p):
    u_hat = 0

    # A
    out = np.array([
            [0, 1, 0, 0],
            [lamda+1-3*u_hat**2, 0, -p.a, 0],
            [0, 0, 0, 1],
            [-p.b, 0, lamda+1-3*u_hat**2, 0]
            ])
    return out

def co_contour_adaptive(c,s,p,m,e,fun):

    c.lambda_steps = 0
    # initialize structures
    out = np.zeros(c.max_pts,dtype=np.complex)
    pre_imag = np.zeros(c.max_pts,dtype=np.complex)

    # Kato basis for first two points
    delh = c.first_step
    h = 0
    lamda = fun(np.linspace(h,delh,c.ksteps+2))
    basis_L,proj_L = c.basisL(c.Lproj,c.L,lamda,s,p,c.LA, 1,c.epsl)
    basis_R,proj_R = c.basisR(c.Rproj,c.R,lamda,s,p,c.RA,-1,c.epsr)

    # compute Evans function for 1st point
    out[0] = c.evans(basis_L[:,:,0],basis_R[:,:,0],lamda[0],s,p,m,e)
    hit_points = c.hit_points

    cnt = 0
    temp = stablab.Struct()
    while h < 1:

        # hold on;
        # plot(real(lambda(end)),imag(lambda(end)),'.k','MarkerSize',18)
        # drawnow;
        #
        # hold on;
        # plot(real(temp.evans),imag(temp.evans),'.k','MarkerSize',18)
        # drawnow;


        temp.evans = c.evans(basis_L[:,:,-1],basis_R[:,:,-1],lamda[-1],s,p,m,e)

        # check relative error of image
        rel_err = abs(temp.evans-out[cnt])/min(abs(temp.evans),abs(out[cnt]))

        # rel_err1 = abs(temp.evans-out(cnt))/min(abs(temp.evans),abs(out(cnt)))
        # rel_err2 = abs(conj(temp.evans)-out(cnt))/min(abs(temp.evans),abs(out(cnt)))
        # rel_err = min(rel_err1,rel_err2)
        # if rel_err2< rel_err1
        # temp.evans = conj(temp.evans)
        # end
        # E = temp.evans
        # Eold = out(cnt)

        if rel_err < c.tol:

            if 'stats' in c:
                if c.stats == 'on':
                    plt.plot(np.real(lamda[-1]),np.imag(lamda[-1]),'.k')
                    plt.show()

            if cnt+1 > c.max_pts:
                raise ValueError('Maximum number of allowed steps exceeded.')

            h += delh
            # pred = (2/3)*c.tol*delh/rel_err;
            pred = (4/3)*delh

            delh = min(min(pred,c.max_step),1-h)

            if hit_points.size != 0:
                if h+delh > hit_points[0]:
                    delh = hit_points[0] - h
                    hit_points = hit_points[1:]

            lamda = fun(np.linspace(h,h+delh,c.ksteps+2))
            basis_L,proj_L = c.basisL(c.Lproj,c.L,lamda,s,p,c.LA,1,c.epsl,
                                      basis_L[:,:,-1],proj_L[:,:,-1])
            basis_R,proj_R = c.basisR(c.Rproj,c.R,lamda,s,p,c.RA,-1,c.epsr,
                                      basis_R[:,:,-1],proj_R[:,:,-1])

            cnt = cnt + 1
            out[cnt] = temp.evans

            # m.options.AbsTol = e.abs_tol*abs(temp.evans);
            # m.options.RelTol = m.options.AbsTol;

        else:
            if 'stats' in c:
                if c.stats == 'on':
                    plt.plot(np.real(lamda[-1]),np.imag(lamda[-1]),'.r')

            delh = 0.5*delh
            if delh < c.min_step_size:
               raise ValueError(
                      'Required delh smaller than minimum step size specified')

            lamda = fun(np.linspace(h,h+delh,c.ksteps+2))
            basis_L,proj_L = c.basisL(c.Lproj,c.L,lamda,s,p,c.LA,1,c.epsl,
                                        basis_L[:,:,0],proj_L[:,:,0])
            basis_R,proj_R = c.basisR(c.Rproj,c.R,lamda,s,p,c.RA,-1,c.epsr,
                                        basis_R[:,:,0],proj_R[:,:,0])

    out = out[0:cnt]
    pre_imag = pre_imag[0:cnt]
    return out, pre_imag

if __name__ == "__main__":

    p = stablab.Struct()
    s = stablab.Struct()

    # parameters
    p.a = 0.1
    p.b = -1

    # study controls
    s.I = 30
    N = 2**7

    # Initialize STABLAB structures
    s.L = -s.I
    s.R = s.I
    a = s.R
    b = 0
    evan_mat = Avec # Avec is the function A where W' = AW
    s.n = 4

    # sets STABLAB structures to their default settings
    s,e,m,c = stablab.emcset(s,'front',
        stablab.evans.LdimRdim(evan_mat,s,p),'reg_reg_polar',evan_mat)

    e.A_pm = A_pm
    c.first_step = 0.05 # first step in h to try
    c.max_step = 0.05 # maximum step in h allowed
    c.min_step_size = 1e-6 # minimum step size in h allowed
    e.abs_tol = 1e-4 # abs tolerance desired in solution
    c.tol = 0.1
    c.max_pts = 5000 # maximum number of points on contour allowed
    c.Lproj = lambda matrix,posneg,eps: stablab.projection5(matrix,posneg,eps,e)
    c.Rproj = lambda matrix,posneg,eps: stablab.projection5(matrix,posneg,eps,e)

    fun = lambda h: 3+np.exp(1j*2*np.pi*h)
    c.hit_points = np.array([])

    m.stats = 'off'
    c.stats = 'on'
    m.eig_tol = 0.01
    m.num_polys = 1
    m.degree = 60
    e.evans = 'reg_reg_bvp_cheb'
    tstart = time.time()
    D_cheb,domain = co_contour_adaptive(c,s,p,m,e,fun)
    tstop = time.time()

    print("\n\n\nRun Time = %f seconds\n\n\n" % (tstop-tstart))
    D_cheb = D_cheb/D_cheb[0]

    stablab.Evans_plot(D_cheb)
