import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
#u.set_enabled_equivalencies(u.dimensionless_angles())
import cosmology as c
#import useful_el as usel
import NFWcris as NFWc
import clusterlensing as cl    
from scipy.integrate import quad
import astropy.constants as con
from scipy import signal

def cm_bhat(mv,zv,cosmo,Delta_C):
    if(Delta_C==200): # full sample, Delta=Delta_{200}
        av = 0.54; bv = 5.9; cv = -0.35;
    else: #  full sample, Delta=Delta_{vir}
        av = 0.9;  bv = 7.7;  cv = -0.29;
    Dz = growth_linear_perturbation(zv,cosmo)
    v = 1./Dz * ( 1.12*(mv/(5e13/cosmo.h))**0.3 + 0.53 )
    cm_bhat = Dz**av*bv*v**cv
    return cm_bhat



def cm_duffy(mv,zv,cosmo,Delta_C):
    """
    % halo virial mass [1e15Msun]
    % halo redshift
    % halo concentration
    """
    
    h0 = cosmo.h
    mpivot = 2.e-3
    
    if Delta_C==200: # full sample (z=0-2), Delta=Delta_{200}
        av = 5.71e0; bv = -0.084e0; cv = -0.47e0;
    #elif Delta_C==500:
        #M200 = cm_conversion(mv,cm_duffy(M200),200,500);
    else: #  full sample (z=0-2), Delta=Delta_{vir}
        av = +7.85e0; bv = -0.081e0;  cv = -0.71e0;
    
    # [1e15Msun]
    mv = mv/(1.e15*u.Msun)
    m0 = mpivot / h0
    c = av * (mv/m0)**bv *(1.+zv)**cv
    return c
######################
    

def m(x):
    m = np.log(1.+x) - x/(1.+x)
    return m
    

def cm_conversion(Mv,cv=0,Dv=200,Dh=500,zv=None,cosmo=None):
    if cv == 0:
        cv = cm_duffy(Mv,zv,cosmo,Dv)
    rv = rvir_NFW(zv,Mv,Delta_C=Dv,cosmo=cosmo)
    rs = rv/cv
    x0 = 1
    x = 0.1
    while np.abs(x/x0-1) > 1e-3:
        x0 = x
        Mh = Mv * m(x*cv)/m(cv)
        x = rvir_NFW(zv,Mh,Delta_C=Dh,cosmo=cosmo) / rv
    rh = x * rv
    ch = rh / rs
    return (Mh, ch)
    
##########################
    
def rvir_NFW(z,Mvir,Delta_C=None,cosmo=None):
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    if Delta_C is None:
        Delta_C = c.Delta_vir(z,cosmo) * cosmo.Om(z)
    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.)
    rvir = (Mvir/(4.*np.pi/3.*rho_cz*Delta_C))**(1./3.)
    return rvir

def MDelta(r,Mr,z,Delta=200,cosmo=None):
    from scipy import interpolate as intr
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    Mdelta = intr.interp1d(Mr / (4*np.pi/3 * cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.) * r**3),Mr)
    return Mdelta(Delta)

def kappa_NFW(x):
    """
    %  kappa = kappa_NFW(x)
    % unitless kappa - need to multiply by convergence scale ks_nfw
    % x - r/rs %dimentionless radius
    """
    x = np.array(x)
    a = np.sqrt(x**2-1.)
    fx = np.arctan(a) / a
    a = np.sqrt(1.-x[x<1.]**2)
    fx[x<1.] = np.arctanh(a) / a
    fx[x==1.] = 1.
    kappa = (1.-fx)/(x**2-1)
    kappa[x==1.] = 1./3
    return kappa

def kappabar_NFW(x):
    """
    %  kappabar = kappabar_NFW(x)
    % unitless kappa - need to multiply by convergence scale ks_nfw
    % x - r/rs %dimentionless radius
    """
    x = np.array(x)
    a = np.sqrt(x**2-1.)
    fx = np.arctan(a) / a
    a = np.sqrt(1.-x[x<1.]**2)
    fx[x<1.] = np.arctanh(a) / a
    fx[x==1.] = 1.
    gx = fx + np.log(0.5 * x) # integral?
    kappabar = 2 * gx / x**2 # mean?
    return kappabar
    


def ks_gnfw(r, z, zs, Mvir, cvir, Delta_C=None, cosmo=None):
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    if Delta_C is None:
        Delta_C = c.Delta_vir(z,cosmo) * cosmo.Om(z)
    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3)
    delta_cvir = c.delta_c(cvir, Delta_C)
    rvir = rvir_NFW(z,Mvir,Delta_C,cosmo)
    rs = rvir/cvir
    ks_gnfw = 2 * rs * delta_cvir * rho_cz / c.Sigma_crit(z,zs,cosmo)    
    return  ks_gnfw
    

def Sigma_NFW_r(r,z,Mvir,cvir,Delta_C=None,cosmo=None):
    """
    %%%%%%%%%
    % sigma=Sigma_NFW(r,z,Mvir,cvir)
    % NFW Sigma in Msun/Mpc^2 in shells of theta
    % (using Shimizu etal 2003?; Wright & Brainerd)
    %%%%%%%%%
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    if Delta_C is None:
        Delta_C = c.Delta_vir(z,cosmo) * cosmo.Om(z)

    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.0)
    delta_cvir = c.delta_c(cvir, Delta_C)
    rvir = (Mvir / (4*np.pi/3*rho_cz*Delta_C)) ** (1./3) # wrt critical density
    rs = rvir/cvir
    x = r/rs
    
    cscale =  2*rs*delta_cvir*rho_cz
    
    kappa = kappa_NFW(x)
    sigma = kappa*cscale
    return sigma


def Sigmabar_NFW_r(r,z,Mvir,cvir,Delta_C=None,cosmo=None):
    """
    %%%%%%%%%
    % sigma=Sigmabar_NFW(r,z,Mvir,cvir)
    % NFW Sigma in Msun/Mpc^2 in shells of theta
    % (using Shimizu etal 2003?; Wright & Brainerd)
    %%%%%%%%%
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    if Delta_C is None:
        Delta_C = c.Delta_vir(z,cosmo) * cosmo.Om(z)
    
    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.0)
    delta_cvir = c.delta_c(cvir, Delta_C)
    rvir = (Mvir / (4.*np.pi/3.*rho_cz*Delta_C)) ** (1./3) # wrt critical density
    rs = rvir/cvir
    x = r/rs
    
    cscale =  2*rs*delta_cvir*rho_cz
    
    kappabar = kappabar_NFW(x)
    sigmabar = kappabar*cscale
    return sigmabar


#%%
# input models that work with mcmc module    
    
def gT_NFW(par,data):
    """ gT_NFW(theta,data)
    tangential shear from NFW model, 
    given model parameters in theta, and data points
    theta =     [Mvir, cvir]
    data = struct containing z, D = Dl*(Dls/Ds) and xdata (radial bins) in arcmin
    return gT
    """
    if len(par) == 4:
        Mvir, cvir, xc, yc = par
    elif len(par) == 3:
        Mvir, xc, yc = par
        cvir = 0
    elif len(par) == 2:
        Mvir, cvir = par
        xc, yc = (0,0)
    elif len(par) == 1:
        Mvir = par[0] 
        cvir = 0
        xc, yc = (0,0)
    else:
        raise NameError('Illegal parameter combination to Sigma_NFW_2D')
    z = data['z']
    zs = data['zs']
    Delta_C = data['Delta']
    cosmo = data['cosmo']
    r = data['xdata']
    xc = xc*r.unit
    yc = yc*r.unit
#    if r.ndim == 2:
#        r = usel.radius(r[0],r[1],[xc,yc])

    Mvir = Mvir*1e15*u.Msun
    if cvir == 0:
        cvir = cm_duffy(Mvir,z,cosmo,Delta_C);

    ks = ks_gnfw(r,z,zs,Mvir,cvir,Delta_C,cosmo)
    #x = np.array(r)[:,np.newaxis] / np.array(rvir_NFW(z,Mvir,Delta_C,cosmo)/cvir)[:,np.newaxis].T
    x = r / (rvir_NFW(z,Mvir,Delta_C,cosmo)/cvir)
    a = np.sqrt(x**2-1)
    fx = np.arctan(a) / a
    a = np.sqrt(1-x[x<1]**2)
    fx[x<1] = np.arctanh(a) / a
    fx[x==1] = 1
    kappa = (1-fx)/(x**2-1)
    kappa[x==1] = 1/3    
    gx = fx + np.log(0.5 * x)
    kappabar = 2 * gx / x**2
    #kappa = np.array(ks)[:,np.newaxis].T * kappa
    #kappa_bar = np.array(ks)[:,np.newaxis].T * kappabar
    kappa = ks * kappa
    kappa_bar = ks * kappabar
    gamma = kappa_bar-kappa
    gt = gamma/(1-kappa)
    return gt

def SigmaT_NFW(par, data):
    """ 1/2D NFW model fitting
    provide data, par structs containing:
    ---------
    % r=data.xdata; (r or [X,Y]) in Mpc
    % z=data.z;
    % Delta_C = data['Delta']
    % cosmo = data['cosmo']
    
    % Mvir=par[0]*1e15
    % cvir=par[1] #def=0
    % Xc=par[2] #def=0
    % Yc=par[3] #def=0
    """
    if len(par) == 4:
        Mvir, cvir, xc, yc = par
    elif len(par) == 2:
        Mvir, cvir = par
        xc, yc = (0,0)
    elif len(par) == 1:
        Mvir = par[0] 
        cvir = 0
        xc, yc = (0,0)
    else:
        raise ValueError('Illegal parameter combination to Sigma_NFW_2D')
        
    z = data['z']
    Delta_C = data['Delta']
    cosmo = data['cosmo']
    Mvir = Mvir*1e15*u.Msun # in units of 10^15 Msun/h
    r = data['xdata']
    xc = xc*r.unit
    yc = yc*r.unit    
#    if r.ndim == 2:
#        r = usel.radius(r[0],r[1],[xc,yc])
    
    if cvir == 0:
        cvir = cm_duffy(Mvir,z,cosmo,Delta_C)
    Sigma = Sigma_NFW_r(r,z,Mvir,cvir,Delta_C,cosmo); #h Msun/Mpc^2
    return Sigma
    
def DSigmaT_NFW(par,data):
    """
    % DSigmaT=DSigmaT_NFW(par,data)
    ------------
    % r=data.xdata; in Mpc
    % z=data.z;
    % Mvir=par(1)*1e15;
    % cvir=par(2);
   """
    if len(par) == 4:
        Mvir, cvir, xc, yc = par
    elif len(par) == 3:
        Mvir, xc, yc = par
        cvir = 0
    elif len(par) == 2:
        Mvir, cvir = par
        xc, yc = (0,0)
    elif len(par) == 1:
        Mvir = par[0] 
        cvir = 0
        xc, yc = (0,0)
    else:
        raise NameError('Illegal parameter combination to Sigma_NFW_2D')
    z = data['z']
    Delta_C = data['Delta']
    cosmo = data['cosmo']
    r = data['xdata']
    xc = xc*r.unit
    yc = yc*r.unit
#    if r.ndim == 2:
#        r = usel.radius(r[0],r[1],[xc,yc])
    
    Mvir = Mvir*1e15*u.Msun # in units of 10^15 Msun/h    
    if cvir == 0:
        cvir = cm_duffy(Mvir,z,cosmo,Delta_C);
        #cvir = cm_bhat(Mvir,z,Cosmo,Delta_C);
    Sigma = Sigma_NFW_r(r,z,Mvir,cvir,Delta_C,cosmo) #h Msun/Mpc^2
    Sigma_bar = Sigmabar_NFW_r(r,z,Mvir,cvir,Delta_C,cosmo) #h Msun/Mpc^2
    DSigmaT = Sigma_bar - Sigma
    return DSigmaT

    
def DSigmaT_NFW_co(par,data):
    """
    % DSigmaT=DSigmaT_NFW_co(par,data)
    comoving
    ------------
    % r=data.xdata; in Mpc
    % z=data.z;
    % Mvir=par(1)*1e15;
    % cvir=par(2);
    """
    if len(par) == 4:
        Mvir, cvir, xc, yc = par
    elif len(par) == 3:
        Mvir, xc, yc = par
        cvir = 0
    elif len(par) == 2:
        Mvir, cvir = par
        xc, yc = (0,0)
    elif len(par) == 1:
        Mvir = par[0] 
        cvir = 0
        xc, yc = (0,0)
    else:
        raise NameError('Illegal parameter combination to Sigma_NFW_2D')

    z = data['z']
    Delta_C = data['Delta']
    cosmo = data['cosmo']
    r = data['xdata']
    xc = xc*r.unit
    yc = yc*r.unit
#    if r.ndim == 2:
#        r = usel.radius(r[0],r[1],[xc,yc])
    
    Mvir = Mvir*1e15*u.Msun # in units of 10^15 Msun/h    
    if cvir == 0:
        cvir = cm_duffy(Mvir,z,cosmo,Delta_C);
        #cvir = cm_bhat(Mvir,z,Cosmo,Delta_C);
    
    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.0)/(1+z)**3 # in Msun/Mpc^3 - this is for comoving
    delta_cvir = c.delta_c(cvir, Delta_C)
    rvir = (Mvir / (4.*np.pi/3.*rho_cz*Delta_C)) ** (1./3.) # wrt critical density
    rs = rvir/cvir
    x = r/rs
    
    cscale =  2.*rs*delta_cvir*rho_cz
    
    kappa = kappa_NFW(x)
    Sigma = kappa * cscale
    kappabar = kappabar_NFW(x)
    Sigma_bar = kappabar * cscale    
    DSigmaT = Sigma_bar - Sigma
    return DSigmaT
    
def DSigmaT_NFW_co_logmass(par,data):
    # turn mass from log:
    par[0] = 10**par[0] 
    DSigmaT = DSigmaT_NFW_co(par,data)
    return DSigmaT

def DSigmaT_NFW_co_logcM(par,data):
    # turn mass from log:
    par2=par[:]
    par2[0] = 10**par[0]
    par2[1] = 10**par[1]
    DSigmaT = DSigmaT_NFW_co(par2,data)
    return DSigmaT


def DSigmaT_NFW_offset_co(par,data):
    """
    % DSigmaT=DSigmaT_NFW_co(par,data)
    comoving
    ------------
    % r=data.xdata; in Mpc
    % z=data.z;
    % Mvir=par(1)*1e15;
    % cvir=par(2);
    """

    
    if len(par) == 3:
        Mvir, cvir, sigma_off = par
    elif len(par) == 2:
        Mvir, sigma_off = par
        cvir = 0
    elif len(par) ==1:
        Mvir = par
        sigma_off = data['roff']
    else:
        raise NameError('Illegal parameter combination to Sigma_NFW_2D')

    z = data['z']
    Delta_C = data['Delta']
    cosmo = data['cosmo']
    r = data['xdata']
    
    Mvir = Mvir*1e15*u.Msun # in units of 10^15 Msun/h  
    sigma_off = sigma_off*u.Mpc
    if cvir == 0:
        cvir = cm_duffy(Mvir,z,cosmo,Delta_C);
        #cvir = cm_bhat(Mvir,z,Cosmo,Delta_C);
    
    rho_cz = cosmo.critical_density(z).to(u.M_sun/u.Mpc**3.)/(1+z)**3 # in Msun/Mpc^3 - this is for comoving
    delta_cvir = c.delta_c(cvir, Delta_C)
    rvir = (Mvir / (4.*np.pi/3.*rho_cz*Delta_C)) ** (1./3.) # wrt critical density
    rs = rvir/cvir
    x = r/rs
    sigma_s =  2.*rs*delta_cvir*rho_cz # in Msun/Mpc^2
    #sigma_s = sigma_s.to(u.Msun/u.pc**2)
    
    
    # for cristobal's offset code'
    
    sigma_x = sigma_off/rs   
    angles = np.linspace(0., 2*np.pi, 180)
    x_range = np.logspace(-5, np.log10(1.01*x.max()), 2**5+1)
    x_range = np.append(0., x_range)
    x = np.append(0.,x)

#    #xvec = np.logspace(np.log10(x.min()),np.log10(x.max()),2**7)
#    P = x_range/sigma_x**2 * np.exp(-0.5*(x_range/sigma_x)**2) # 2D Gaussian distribution
    P = np.zeros(x_range.size)
    P[np.argmin(np.abs(x_range-sigma_x))]=1 # simple delta function for testing

    print "x: %s " % x
    print "x_range: %s " % x_range
    print "P: %s " % P
    print "simga_s: %s " % sigma_s
    print "rs: %s " % rs



    DSigmaT = NFWc.esd_offset(x, xoff=x_range, n=P, sigma_s=sigma_s/2, x_range=x_range, angles=angles, interp_kind='slinear')    
    DSigmaT = (DSigmaT*u.Msun/u.pc**2).to(u.Msun/u.Mpc**2) 
    return DSigmaT



    
def DSigmaT_NFW_plusoffset_co(par,data):
    """
    % DSigmaT=DSigmaT_NFW_co(par,data)
    comoving
    ------------
    % r=data.xdata; in Mpc
    % z=data.z;
    % Mvir=par(1)*1e15;
    % cvir=par(2);
    """
    if len(par) == 4:
        Mvir, cvir, roff, alpha = par
    elif len(par) == 3:
        Mvir, roff, alpha = par
        cvir = 0
    elif len(par) == 1:
        Mvir = par
        cvir = 0
        roff = data['roff']
        alpha = data['alpha']
    else:
        raise ValueError('Illegal parameter combination to DSigma_NFW_2halo')
    if cvir == 0:
        cvir = 'Dutton' # using clusterlensing module
    else:
        cvir = [cvir]

    c = cl.ClusterEnsemble([data['z']],cosmology=data['cosmo'],cm=cvir)  
    c.m200 = [Mvir*1e15]*u.Msun
    c.calc_nfw(data['xdata'], offsets=[roff]*u.Mpc,numRoff=50,numRinner=5,numTh=90,factorRouter=1) # run
    dsigma_nc = c.deltasigma_nfw.flatten() # with offset
    c.calc_nfw(data['xdata']) # run
    dsigma_c = c.deltasigma_nfw.flatten()
    dsigma = (1-alpha) * dsigma_c + alpha * dsigma_nc
   
    return dsigma.to(u.Msun/u.Mpc**2)


def DSigmaT_NFW_co_cl(par,data):
    """
    % DSigmaT=DSigmaT_NFW_co(par,data)
    comoving
    using clusterlensing module for consistency check
    ------------
    % r=data.xdata; in Mpc
    % z=data.z;
    % Mvir=par(1)*1e15;
    % cvir=par(2);
    """
    if len(par) == 2:
        Mvir,cvir = par
    elif len(par) == 1:
        Mvir = par
        cvir = 0
    else:
        raise ValueError('Illegal parameter combination to DSigma_NFW_2halo')
    if cvir == 0:
        cvir = 'Dutton' # using clusterlensing module
    else:
        cvir = [cvir]

    c = cl.ClusterEnsemble([data['z']],cosmology=data['cosmo'],cm=cvir)  
    c.m200 = [Mvir*1e15]*u.Msun
    c.calc_nfw(data['xdata']) # run
    dsigma_c = c.deltasigma_nfw.flatten()
   
    return dsigma_c.to(u.Msun/u.Mpc**2)


#%% pressure profiles, equations taken from Planck 2013 pressure profile paper,  A&A, 550 (2013) A131

def P500(z,M500,cosmo):
    """ P500 scales with cluster mass in a standard self-similar model
    Equation (10)
    """
    h70 = cosmo.h/0.7
    return 1.65e-3 * cosmo.efunc(z)**(8./3) * (M500/(3e14*u.Msun/h70))**(2./3)  * h70**2 * (u.keV * u.cm**(-3))
    
def Px_gNFW(x, logP0, c500):
    """ Pressure profile, normalized, scaled. 
    Equation (7)
    """
    gamma, alpha, beta = [0.308, 1.05, 5.49] # Planck set parameters
    P0 = 10**logP0 # amplitude
    return P0/( (c500*x)**gamma * (1+(c500*x)**alpha)**((beta-gamma)/alpha) )

def Pr(r,z,M500,logP0,c500,cosmo=None,Dv=500):
    """ Pressure profile, comoving
    Equation (11)
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    h70 = cosmo.h/0.7
    R500 = rvir_NFW(z,M500,Delta_C=Dv,cosmo=cosmo)
    x = r/R500
    nss_factor = (M500/(3e14*u.Msun/h70))**0.12 # non-self-similar factor as given by Eq (11); check w/Nick
    return P500(z,M500,cosmo) * Px_gNFW(x,logP0,c500) * nss_factor
    
def ySZ_r(r,z,M500,logP0,c500, cosmo=None,Dv=500): 
    """SZ Compton y profile
    project pressure in a cylinder
    Equation (2)
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)

    # here come's the integration part. 
    # eq 2 says to integrate from r to Rb. What's Rb?
    R500 = rvir_NFW(z,M500,Delta_C=Dv,cosmo=cosmo)
    Rmax = 5*R500
        
    # r is an array.  we need to break this up to scalars for quad
    integral = np.zeros(r.shape)
    for (i,r_i) in enumerate(r): 
        # quad doesn't seem to deal with quantities, so stripping it down
        integral[i] = quad(lambda rr: 
            2. * Pr(rr*u.Mpc,z,M500,logP0,c500,cosmo,Dv).value  
            * rr /np.sqrt(rr**2 - r_i.value**2), 
            r_i.value, Rmax.value)[0] # what are the integral limits? changed upper to 5*R500
    unit = u.keV/u.cm**2 # units of integral
    integral = integral*unit # quad doesnt work well with units (?), this restores unit

    factor = con.sigma_T.to(u.cm**2) / (con.m_e*con.c**2).to(u.keV) # 
    y = factor * integral # now y is dimensionless
    return y


def ySZ_convolved(r,z,M500,logP0,c500, fwhm_beam, cosmo=None,Dv=500):
    """ 
    observed y profile - Equation (3)
    real y is convolved with the beam. when done in angular units it's simple; 
    when done is physical, should we just convert psf size from arcmin to Mpc?
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    # convert the beam from angular (theta) to comoving (r). This is likely wrong but what's been used. 
    psf = fwhm_beam/ np.sqrt(8.*np.log(2.)) * u.arcmin # in arcmin
    psf_r = (psf*cosmo.kpc_comoving_per_arcmin(z)).to(u.Mpc) # in Mpc
    # made a gaussian filter with beam width
    gauss_win = signal.gaussian(51, std=psf_r.value) 
    # make the y profile
    y = ySZ_r(r,z,M500,logP0,c500, cosmo=cosmo,Dv=Dv)
    # convolve y and beam. convolution is done unitless
    y_filtered = signal.convolve(y, gauss_win, mode='same') / sum(gauss_win)
    return y_filtered

   