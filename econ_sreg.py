import numpy as np
import pandas as pd
import econ_stat as stat
import matplotlib.pyplot as plt
from termcolor import colored
from scipy import stats as scp

## Simple Regression
def slope(x,y):
    return stat.cov(x.array,y.array)/stat.cov(x.array,x.array)

def itc(x,y):
    return y.mean() - slope(x,y)*x.mean()

def err(x,y):
    if(x.size() != y.size()):
        print("The two data sets are not of the same size!")
        return 0
    e = []
    for i in range(x.size()):
        e.append(y[i] - itc(x,y) - slope(x,y)*x[i])
    return e

def R2(x,y):
    return stat.cov(x.array, y.array)**2 / (stat.cov(x.array, x.array) * stat.cov(y.array, y.array))

def s(x,y):
    e = np.array(err(x,y))
    e2 = e**2
    s = sum(e2)
    return np.sqrt(s/(x.size()-2))

def s2(x,y):
    e = np.array(err(x,y))
    e2 = e**2
    s = sum(e2)
    return s/(x.size()-2)

def sb(x,y):
    return s(x,y)/(np.sqrt(x.size()-1)*x.dev())

def tb(x,y,beta=0):
    return (slope(x,y)-beta)/sb(x,y)

def sa(x,y):
    s1 = (x.size()-1)*x.dev()**2
    return s(x,y)*np.sqrt(1/(x.size()) + (x.mean()**2/s1))

def ta(x,y, alpha):
    return (itc(x,y)-alpha)/sa(x,y)

def SST(x,y):
    return y.var()*(y.size()-1)

def SSE(x,y):
    return x.var()*(x.size()-1)*slope(x,y)**2

def SSR(x,y):
    e = np.array(err(x,y))
    ee = e**2
    return sum(ee)

def line(x,y, prej=0.9, proj=1.1):
    b = slope(x,y)
    a = y.mean() - b*x.mean()
    xx = np.linspace(prej*np.min(x.array), proj*np.max(x.array), 100)
    text = ''.join(('y = %.2f' % b, 'x + %.2f' % a))
    plt.text(0.8*np.max(x.array), 0.95*np.min(y.array), text)
    plt.plot(xx, b*xx + a, color='gray')

def t2b(x, y, beta0=0, alpha=0.05, show=True):
    b = slope(x,y)
    k = scp.t.ppf(1-alpha/2, x.size()-2)
    c = k*sb(x,y)
    t = (b-beta0)/(c/k)
    if show:
        print("We test", colored("H0: beta =", 'magenta'), colored(beta0, 'magenta'),
            "against", colored("H1: beta =/=", 'magenta'), colored(beta0, 'magenta'))
        print("\nReject H0 if the estimated slope is in:\n[-inf, ", beta0-c,
             "] or [", beta0+c, ", inf]", sep='')
        print("\nThe estimated slope equals", b)
    if(b < beta0-c or b > beta0+c):
        if show: print("Therefore, H0", colored("is rejected.", 'magenta'))
        return [t, b, True]
    else:
        if show: print("Therefore, H0", colored("cannot be rejected.", 'magenta'))
        return [t, b, False]


def t2a(x, y, alpha0, alpha=0.05, show=True):
    a = itc(x,y)
    k = scp.t.ppf(1-alpha/2, x.size()-2)
    c = k*sa(x,y)
    if show:
        print("We test", colored("H0: alpha =", 'magenta'), colored(alpha0, 'magenta'),
            "against", colored("H1: alpha =/=", 'magenta'), colored(alpha0, 'magenta'))
        print("\nReject H0 if the estimated intercept is in:\n[-inf, ", alpha0-c,
             "] or [", alpha0+c, ", inf]", sep='')
        print("\nThe estimated intercept equals", a)
    if(a < alpha0-c or a > alpha0+c):
        if show: print("Therefore, H0", colored("is rejected.", 'magenta'))
    else:
        if show: print("Therefore, H0", colored("cannot be rejected.", 'magenta'))
    return k

def confb95(x,y, show=True):
    c = t2b(x,y, x.size()-2, slope(x,y), show=False) * sb(x,y)
    if show:
        print("From the t-test we conclude the confidence interval for the slope:")
        print("[", slope(x,y)-c, ", ", slope(x,y)+c, "]", sep='')
    return [slope(x,y)-c, slope(x,y)+c]

def confa95(x,y, show=True):
    c = t2a(x,y, x.size()-2, itc(x,y), show=False) * sa(x,y)
    if show:
        print("From the t-test we conclude the confidence interval for the intercept:")
        print("[", itc(x,y)-c, ", ", itc(x,y)+c, "]", sep='')
    return [itc(x,y)-c, itc(x,y)+c]