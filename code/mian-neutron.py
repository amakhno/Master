import numpy as np
from scipy.special import gamma


def DF_INI(Eps, l, cl, RMAX, R_INI):
    NDIF = 401
    cl_ = cl
    l_ = l
    Eps_ = Eps
    pi = np.pi
    mu = 1.0*A/(A+1.0)*1.01/5.485803e-4
    k = np.sqrt(2.0*mu*Eps)
    if (cl == 1):
        o_r0 = (1.322-7.6e-4*A+4.0*A*A*1.0e-6-8.0e-9*A*A*A)*1.0e-13/3.862e-11
    if (cl == 2):
        o_r0 = 1.17*1.0e-13/3.862e-11
    period = 2.0*pi/k
    if (period > o_r0):
        ra = period
    else:
        ra = o_r0
    r0 = ra/6.0e2
    hstep = (R_MAX-r0)/NDIF
    Y1 = []
    Y1.append(r0**(l+1))
    Y1.append(r0**(l+1))
    Y1.append((l+1)*r0**l)
    Y1.append((l+1)*r0**l)

    ido=1; rini=r0

    for ISTEP in range(1, NDIF):
        r=rini+hstep
        R_INI(ISTEP)=r
        PARAM = np.array(50)


pi = np.pi
alfa = 1.0/137.0360

A = 106
Z = 46
N = A-Z
mu = 1.0*A/(A+1.0)*1.01/5.485803e-4
R0 = 8.0*1.2*A**(1.0/3.0)*1.0e-13/3.862e-11
dd = 2.5/0.511
gv = 2.9899e-12

RNUC = 1.2*A**(1.0/3.0)*1.0-13/3.862e-11

for IEi in range(5, 5):
    eps_i = (3.0+IEi)/0.511
    k_i = np.sqrt(2.0*mu*eps_i)
    AEpsf = 0
    BEpsf = eps_i-dd-1.0  # /0.511
    MEpsf = 5
    HEpsf = (BEpsf-AEpsf)/2.0/MEpsf

    for IEpsf in range(1, 2*MEpsf-1):
        eps_f = AEpsf+IEpsf*HEpsf

        AEe = 1.0
        BEe = eps_i-dd-eps_f
        MEe = 8
        HEe = (BEe-AEe)/2.0/MEe

        MULEe = 1
        INTEe = 0.0

        for IEe in range(1, 2*MEe-1):
            E_e = AEe+IEe*HEe
            k_e = np.sqrt(E_e**2.0-1.0)
            k_nu = eps_i-eps_f-E_e-dd

            del1 = np.sqrt(1.0-(alfa*(Z+1))**2)
            nn = alfa*(Z+1)*E_e/k_e
            argg = del1+nn*1j
            F_F = 2.0*(1.0+del1)*(2.0*k_e*RNUC)**(2.0*(del1-1.0))*np.exp(pi*nn)
            Fun_F = F_F*np.abs(del1+nn*1j)**2/(gamma(2.0*del1+1.0))**2

            if (np.abs(k_nu/(eps_i-eps_f)) >= 1e-14):
                GOTO 35

            SLF = 0.0
            for l_f in range(0, 2):
                SLI = 0.0
                for l_i in range(0, 2):
