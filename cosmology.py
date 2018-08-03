import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as con
from astropy import units as u
#from itertools import izip, coun
from scipy.interpolate import interp1d
from scipy.integrate import quad


def Delta_vir(z,cosmo=None):
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    wf = (1/cosmo.Om(z)-1)
    Delta_vir = 18. * np.pi**2. * (1 + 0.40929 * wf**0.90524)
    return Delta_vir

def delta_c(cvir, Delta_C):
    delta = Delta_C/3. * cvir**3. / ( np.log(1.+cvir) - cvir/(1.+cvir) )
    return delta

    
def DlsDs(zl,zs,cosmo=None):
    #zl = np.array(zl,dtype=np.float64); zs = np.array(zs,dtype=np.float64)
    if zl.size==1:
        zl = zl * np.ones(zs.shape)
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    Ds = cosmo.angular_diameter_distance(zs)
    Dls = np.zeros(zs.shape)
    Ibg = (zl<zs) & ~(np.isclose(zl,zs)) # to make Dls zero
    Dls[Ibg] = cosmo.angular_diameter_distance_z1z2(zl[Ibg],zs[Ibg])
    return Dls/Ds
    
def fchi_tab(zmax,cosmo):
    zx = 10.**np.linspace(-4, np.log10(zmax),100)
    chi = np.zeros_like(zx)
    for (i,z) in enumerate(zx):
        chi[i] = quad(lambda zz: (con.c.to(u.km/u.s)/cosmo.H0).value /np.sqrt((1+zz)**3*cosmo.Om0+cosmo.Ode0), 0., z)[0]
    zx = np.append(0., zx)
    chi = np.append(0., chi)
    fchi = interp1d(zx, chi, kind="linear",fill_value="extrapolate")
    return fchi
    
def DlsDs_int(zl,zs,cosmo=None):
    if zl.size==1:
        zl = zl * np.ones(zs.shape)
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)

    fchi = fchi_tab(np.max(np.append(zs,zl)),cosmo) #comv distance lookup table   
    Dl = fchi(zl)
    beta = 1.-Dl/fchi(zs)
    beta[beta<0.] = 0.
    return beta

def D(zl,zs,cosmo=None):
    #zl = np.array(zl,dtype=np.float64); zs = np.array(zs,dtype=np.float64)
    if zl.size==1:
        zl = zl * np.ones(zs.shape)
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3,)
    fchi = fchi_tab(np.max(np.append(zs,zl)),cosmo) #comv distance lookup table   
    Dl = fchi(zl)/(1+zl)
    dlsds = DlsDs_int(zl,zs,cosmo=cosmo)
    return 1./(Dl*dlsds)


def Sigma_crit(zl,zs,cosmo=None):
    #c = const.c.to(u.Mpc/u.s) 
    #G = const.G.to(u.Mpc**3/u.M_sun/u.s**2)
    c = 9.7156119e-15*(u.Mpc/u.s)
    G = 4.5183594e-48*(u.Mpc**3/u.Msun/u.s**2)
    Dlens = D(zl,zs,cosmo) # Ds/Dls/Dl lensing ratio
    Sig_crit = c**2 / (4*np.pi*G) * Dlens
    return Sig_crit
    
def Sigma_crit_co(zl,zs,cosmo=None):
    return Sigma_crit(zl,zs,cosmo=cosmo)/(1.+zl)**2
    
def zsource_D(D,zl,cosmo=None):
    zl=np.array(zl); D = np.array(D)
    zs = np.linspace(1e-5,5,1000)
    if zl.size==1:
        zl = zl * np.ones(zs.shape)
    dlsds = DlsDs(zl, zs, cosmo=cosmo)
    zD = np.interp(D,dlsds,zs)   # z(<D>)
    return zD
    
def mean_dlsds(zl,zs,cosmo=None):
    DlsDs(zl,zs,cosmo=cosmo)
    
def Sigma_cr_PDF(zl, Pz, zvec=None, cosmo=None): # not working
    c = 9.7156119e-15*(u.Mpc/u.s)
    G = 4.5183594e-48*(u.Mpc**3/u.Msun/u.s**2)
    sigma_cr_fact = c**2 / (4*np.pi*G)
    if zvec is None:
        zvec = np.arange(0,7.01,0.01)
    zmat = np.tile(zvec,[len(zl),1])
    ad_zs= cosmo.angular_diameter_distance(zvec)
    ad_zs_m = np.tile(ad_zs,[len(zl),1])
    zlmat = np.tile(zl,[len(zvec),1]).T
    ad_zl = cosmo.angular_diameter_distance(zl)
    Pzmask = np.ma.array( data=Pz, mask = (zmat<=zlmat) )
    zmatmask = np.ma.array( data=zmat, mask = (zmat<=zlmat) )
    ad_zls_m = cosmo.angular_diameter_distance_z1z2(zlmat,zmatmask)
    Sigma_cr = sigma_cr_fact * (np.ma.sum( (ad_zs_m.value/ad_zls_m.value)*Pzmask, axis=1).data / np.sum(Pz,axis=1)) / ad_zl
    import gc
    del zmat, zlmat,  Pzmask   
    gc.collect()
    return Sigma_cr

def Sigma_cr_co_PDF(zl, Pz, zvec=None, cosmo=None):
    """ calculate comoving Sigma_cr using full P(z) integration """
    c = 9.7156119e-15*(u.Mpc/u.s)
    G = 4.5183594e-48*(u.Mpc**3/u.Msun/u.s**2)
    sigma_cr_fact = c**2 / (4*np.pi*G) # physical factors
    if zvec is None: #default
        zvec = np.arange(0,7.01,0.01)
    zmat = np.tile(zvec,[len(zl),1]) # source redshift matrix
    chi_zs= cosmo.comoving_distance(zvec) #source comoving distance 
    chi_zs_m = np.tile(chi_zs.value,[len(zl),1]) #source comoving distance matrix
    zlmat = np.tile(zl,[len(zvec),1]).T # lens redshift matrix
    chi_zl = cosmo.comoving_distance(zl) # lens comoving ditance
    ad_zl = cosmo.angular_diameter_distance(zl) # lens angular diameter distance 
    chi_zl_m = np.tile(chi_zl.value,[len(zvec),1]).T # lens comoving distance matrix
    Pzmask = np.ma.array( data=Pz, mask = ( (zmat<=zlmat) | (np.isclose(zmat,zlmat)) ) ) # P(z) masked below lens redshift
    Sigma_cr = sigma_cr_fact / (np.ma.sum( (1 - chi_zl_m/chi_zs_m)*Pzmask, axis=1).data / np.sum(Pz,axis=1)) / ad_zl / (1+zl)**2 # Sigma_cr calculation masking below lens redshift 
    import gc
    del zmat, zlmat, chi_zl_m, Pzmask, chi_zs_m    
    gc.collect()
    return Sigma_cr

def Sigma_cr_co_PDF_lookup(zl, Pz, zvec=None, cosmo=None):
    """ calculate comoving Sigma_cr using full P(z) integration, with interpolation for distances  """
    c = 9.7156119e-15*(u.Mpc/u.s)
    G = 4.5183594e-48*(u.Mpc**3/u.Msun/u.s**2)
    sigma_cr_fact = c**2 / (4*np.pi*G) # physical factors
    if zvec is None: #default
        zvec = np.arange(0,7.01,0.01)
    fchi = fchi_tab(np.max(np.append(zl,zvec)),cosmo) #comv distance lookup table   
    zmat = np.tile(zvec,[len(zl),1]) # source redshift matrix
    chi_zs= fchi(zvec)*u.Mpc #source comoving distance 
    chi_zs_m = np.tile(chi_zs.value,[len(zl),1]) #source comoving distance matrix
    zlmat = np.tile(zl,[len(zvec),1]).T # lens redshift matrix
    chi_zl = fchi(zl)*u.Mpc # lens comoving ditance
    chi_zl_m = np.tile(chi_zl.value,[len(zvec),1]).T # lens comoving distance matrix
    Pzmask = np.ma.array( data=Pz, mask = ( (zmat<=zlmat) | (np.isclose(zmat,zlmat)) ) ) # P(z) masked below lens redshift
    Sigma_cr = sigma_cr_fact / (np.ma.sum( (1 - chi_zl_m/chi_zs_m)*Pzmask, axis=1).data / np.sum(Pz,axis=1)) / chi_zl / (1+zl) # Sigma_cr calculation masking below lens redshift
    import gc
    del zmat, zlmat, chi_zl_m, Pzmask, chi_zs_m    
    gc.collect()
    return Sigma_cr


    
def MY(Y,z,cosmo):
    """ M(Y); Y in units of arcmin^2; M up to bias factor"""
    alpha = 1.79
    beta = 2./3.
    Ystar = 10**(-0.19)
    b=0.2
    M =  ( cosmo.efunc(z)**(-beta)*(cosmo.kpc_proper_per_arcmin(0.3).to(u.Mpc/u.arcmin))**2.*Y/(0.0001*u.Mpc**2) /(Ystar * (cosmo.h/0.7)**(-2+alpha)) )**(1/alpha) /(1-b)* 6e14 * u.Msun
    return M

def Msz_Lambda(Lam):
    a = 4.572
    alpha = 0.965
    lnMsz = (np.log(Lam) - a)/alpha
    Msz = np.exp(lnMsz) * 5.23e14
    return Msz
    
def comoving_histogram(z,cosmo,bins=10, plot=True, **kwargs):
    nz,zbins = np.histogram(z, bins=bins )
    dV =  cosmo.comoving_volume(zbins)
    dVdiff = np.diff(dV)
    nz = nz/dVdiff
    if plot:
        import matplotlib.pyplot as plt
        plt.subplots_adjust(hspace=0.001,bottom=0.15,left=0.2)
        plt.bar(zbins[:-1],nz.value,width=zbins[1]-zbins[0],**kwargs)
        plt.xlabel('Redshift')
        plt.ylabel(r'Comoving Number Density [$h^3$ Mpc$^{-3}$]')
    return zbins, nz
  
    