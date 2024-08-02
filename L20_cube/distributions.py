import scipy
from scipy.special import legendre
import numpy as np
from math import cos

def firstpassage(tt, theta0):
    n = np.arange(100)
    aa = [2*scipy.special.eval_legendre(2*n+1, cos(theta0))*scipy.special.gamma(n+3/2)*(-1)**n*(4*n+3)/(
        np.sqrt(np.pi)*scipy.special.gamma(n+1))*np.exp(-(2*n+1)*(2*n+2)*tt)]
    n = np.arange(100, 2000)
    bb = [2*scipy.special.eval_legendre(2*n+1, cos(theta0))*(-1)**n*(4*n+3)*(np.sqrt(n)+3/(
        8*np.sqrt(n)))/(np.sqrt(np.pi))*np.exp(-(2*n+1)*(2*n+2)*tt)]
    return np.sum(aa)+np.sum(bb)


def wald(tt, theta0):
    return (np.pi/2-theta0)/np.sqrt(4*np.pi*tt**3)*np.exp(-(np.pi/2-theta0)**2/(4*tt))
