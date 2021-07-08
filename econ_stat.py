import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab
from termcolor import colored
from scipy import stats as scp

class dataset:
    def __init__(self, data, name=''):
        self.array = data
        self.name = name
    def __getitem__(self, key):
        return self.array[key]
    def __setitem__(self, key, value):
        self.array[key] = value
    def __str__(self):
        return str(self.array)
    def mean(self):
        return np.mean(self.array)
    def median(self):
        return median(self.array)
    def dev(self):
        return dev(self.array)
    def var(self):
        return np.var(self.array)
    def skew(self):
        return skew(self.array)
    def kurt(self):
        return kurt(self.array)
    def size(self):
        return len(self.array)
    def append(self, element):
        self.array.append(element)
    def sort(self, inplace=True):
        if inplace:
            self.array = np.sort(self.array)
        else: return np.sort(self.array)


## Basic Functions
def rounds(x, n):
    if x == 0:
        return x
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))

def mean(data):
    sum = 0

    for x in data:
        sum += x

    return sum/len(data)

def median(data):
    sorted = np.sort(data)

    lindex = int(np.floor((len(sorted)+1)/2))
    rindex = int(np.ceil((len(sorted)+1)/2))

    return (sorted[lindex-1]+sorted[rindex-1])/2

def var(data):
    N = len(data)/(len(data)-1)
    return N*np.var(data)

def dev(data):
    return np.sqrt(var(data))

def moment(data, r):
    N = 1/(len(data))
    sum = 0
    for i in range(0, len(data)):
        sum += (data[i] - mean(data))**r

    return N*sum

def stand_moment(data, r):
    return moment(data, r)/dev(data)**r

def skew(data):
    return stand_moment(data, 3)

def kurt(data):
    return stand_moment(data, 4)

def ekurt(data):
    return stand_moment(data, 4) - 3

def cov(X,Y):
    return np.cov(X, Y)[0,1]

def cor(X,Y):
    return (cov(X,Y)/(dev(X)*dev(Y)))

def corlist(variables):
    M = np.cov([list(x) for x in variables])
    print(list(M), '\n')
    cors = np.zeros(shape=(len(variables),len(variables)))
    for i in range(len(variables)):
        for j in range(len(variables)):
            cors[i][j] = M[i][j]/np.sqrt(M[i][i]*M[j][j])

    print(cors)


## Exporting Stats
def stats(data):
    return [mean(data), median(data), dev(data), skew(data), kurt(data)]

def show_stats(datavector):
    print("\n\n\t\tMean\t\tMedian\t\tDeviation\t\tSkewness\t\tKurtosis")
    for i in range(0, len(datavector)):
        print(datavector[i].name,
        '\t   ',rounds(datavector[i].mean(), 3),
        '\t\t',rounds(datavector[i].median(), 3),
        '\t\t',rounds(datavector[i].dev(), 3),
        '\t\t',rounds(datavector[i].skew(), 3),
        '\t\t',rounds(datavector[i].kurt(), 3))

def stats2xl(datavector, directory, labels=False):
    means = []
    medians = []
    devs = []
    skews =[]
    kurts = []
    names = []

    for i in range(0, len(datavector)):
        if labels == True:
            names.append(datavector[i].name)
        means.append(datavector[i].mean())
        medians.append(datavector[i].median())
        devs.append(datavector[i].dev())
        skews.append(datavector[i].skew())
        kurts.append(datavector[i].kurt())

    if labels == True:
        DD = {'': names,
            'Mean': means,
            'Median': medians,
            'Deviation': devs,
            'Skewness': skews,
            'Kurtosis': kurts}
        df = pd.DataFrame(DD, columns=['', 'Mean', 'Median', 'Deviation', 'Skewness', 'Kurtosis'])
    else:
        DD = {'Mean': means,
            'Median': medians,
            'Deviation': devs,
            'Skewness': skews,
            'Kurtosis': kurts}
        df = pd.DataFrame(DD, columns=['Mean', 'Median', 'Deviation', 'Skewness', 'Kurtosis'])

    df.to_excel(directory, index = (not labels))

    print('\n >> Excel file has been created at', directory)

## Plotting
def plot(x, y, color='#e36b73', xl=True, yl=True, xlab='', ylab=''):
    if xl:
        if(xlab==''):
            plt.xlabel(x.name)
        else: plt.xlabel(xlab)
    if yl:
        if(ylab==''):
            plt.ylabel(y.name)
        else: plt.ylabel(ylab)
    plt.plot(x.array, y.array, color=color)

def scatter(x, y, color='#e36b73', size=10, xl=True, yl=True, xlab='', ylab=''):
    if xl:
        if(xlab==''):
            plt.xlabel(x.name)
        else: plt.xlabel(xlab)
    if yl:
        if(ylab==''):
            plt.ylabel(y.name)
        else: plt.ylabel(ylab)
    plt.scatter(x.array, y.array, color=color, s=size)

def hist(x, bins=0, alpha=1, color='Pink', histtype='bar', ec='#695969', density=False, labels=True, stats=False):
    if labels:
        plt.xlabel(x.name)
        if density:
            plt.ylabel('Density')
        else: plt.ylabel('Frequency')
    if bins==0:
        B = []
        T = int(np.ceil(np.sqrt(x.size())))
        for i in range(0, T):
            B.append(min(x) + i*((max(x)-min(x))/T))
        plt.hist(x.array, alpha=alpha, bins=B, color=color, histtype=histtype, ec=ec, density=density)
    else: plt.hist(x.array, alpha=alpha, bins=bins, color=color, histtype=histtype, ec=ec, density=density)
    if stats:
        textstr = '\n'.join((
        r'Mean: %.2f' % (x.mean(), ),
        r'Median: %.2f' % (x.median(), ),
        r'Deviation: %.2f' % (x.dev(), )))
        plt.text(0.95*np.max(x.array), 1.25, textstr)

def qq(x, y):
    #Calculate quantiles
    x.sort()
    quantile_levels1 = np.arange(len(x),dtype=float)/len(x)

    y.sort()
    quantile_levels2 = np.arange(len(y),dtype=float)/len(y)

    #Use the smaller set of quantile levels to create the plot
    quantile_levels = quantile_levels2

    #We already have the set of quantiles for the smaller data set
    quantiles2 = y

    #We find the set of quantiles for the larger data set using linear interpolation
    quantiles1 = np.interp(quantile_levels,quantile_levels1,x)

    #Plot the quantiles to create the qq plot
    scatter(dataset(quantiles1), dataset(quantiles2))

    #Add a reference line
    maxval = max(x[-1],y[-1])
    minval = min(x[0],y[0])
    pylab.plot([minval,maxval],[minval,maxval],'k-')

    pylab.show()

## Hypothesis Testing
def t2(data, df, u0, show=True, count=False):
    K = [12.076, 4.303, 3.182, 2.276, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]
    for x in np.linspace(2.201, 1.980, 50):
        K.append(x)
    if df <= 60:
        k = K[df-1]
    else: k = 1.960
    c = k*data.dev()/np.sqrt(df+1)
    if show:
        print("We test", colored("H0: u0 =", 'magenta'), colored(u0, 'magenta'),
            "against", colored("H1: u0 =/=", 'magenta'), colored(u0, 'magenta'))
        print("\nReject H0 if the mean of the data lies in:\n[-inf, ", u0-c,
             "] or [", u0+c, ", inf]", sep='')
        print("\nThe sample mean equals", data.mean())
    if(data.mean() < u0-c or data.mean() > u0+c):
        if show: print("Therefore, H0", colored("is rejected.", 'magenta'))
        if count: return 1
    else:
        if show: print("Therefore, H0", colored("cannot be rejected.", 'magenta'))
        if count: return 0

def t2count(N, datavector, df, u0):
    sum = 0
    for i in range(N):
        sum += t2(datavector[i], df, u0, show=False, count=True)
    return sum

def binom2samples(X1, X2, alpha=0.05, show= True):
    n = len(X1)
    m = len(X2)
    p1 = np.mean(X1)
    p2 = np.mean(X2)
    p0 = (sum(X1)+sum(X2))/(n+m)
    z = (p1-p2)/np.sqrt(p0*(1-p0)*(1/n + 1/m))
    k = scp.norm.ppf(1-alpha/2)
    pvalue = 2*scp.norm.sf(abs(z))

    if show:
        print("We test", colored("H0: p1-p2 = 0", 'magenta'),
            "against", colored("H1: p1-p2 =/= 0", 'magenta'))
        print("\nReject H0 if the z-statistic lies in:\n[-inf, ", -k,
            "] or [", k, ", inf]", sep='')
        print("\nThe z-statistic equals", z)
        print("The p-value equals", pvalue)
    if(z < -k or z > k):
        if show: print("Therefore, H0", colored("is rejected.", 'magenta'))
        return z
    else:
        if show: print("Therefore, H0", colored("cannot be rejected.", 'magenta'))
        return z

def t2samples(X1, X2, alpha=0.05, show = True, count = False):
    simvar = (0.5 < np.std(X1)/np.std(X2) and np.std(X1)/np.std(X2) < 2)
    if simvar:
        print("Using the 2 independent sample t-test for similar variances. (", np.std(X1)/np.std(X2), ")")
        n = len(X1)
        m = len(X2)
        s21 = np.var(X1)
        s22 = np.var(X2)
        sp = np.sqrt(((n-1)*s21 + (m-1)*s22)/(n+m-2))
        t = (np.mean(X1)-np.mean(X2)) / (sp*np.sqrt(1/n + 1/m))
        df = n+m-2
        k = scp.t.ppf(1-alpha/2, df)
        pvalue = 2*scp.t.sf(abs(t),df)

        if show:
            print("We test", colored("H0: u1 = u2", 'magenta'),
                "against", colored("H1: u1 =/= u2", 'magenta'))
            print("\nReject H0 if the t-statistic lies in:\n[-inf, ", -k,
                "] or [", k, ", inf]", sep='')
            print("\nThe t-statistic equals", t)
            print("The p-value equals", pvalue)
        if(t < -k):
            if show: print("\nTherefore, H0", colored("is rejected.", 'magenta'))
            if count: return 1
            else: return t
        else:
            if show: print("\nTherefore, H0", colored("cannot be rejected.", 'magenta'))
            if count: return 0
            else: return t
    else:
        print("Using the 2 independent sample t-test for unequal variances. (", np.std(X1)/np.std(X2), ")")
        n = len(X1)
        m = len(X2)
        s21 = np.var(X1)
        s22 = np.var(X2)
        sz = np.sqrt(s21/n + s22/m)
        t = (np.mean(X1)-np.mean(X2))/sz
        p = s21/n+ s22/m
        q = (s21/n)**2/(n-1)
        r = (s22/m)**2/(m-2)
        df = (p**2)/(q+r)
        k = scp.t.ppf(1-alpha/2, df)
        pvalue = 2*scp.t.sf(abs(t),df)

        if show:
            print("We test", colored("H0: u1 = u2", 'magenta'),
                "against", colored("H1: u1 =/= u2", 'magenta'))
            print("\nReject H0 if the t-statistic lies in:\n[-inf, ", -k,
                "] or [", k, ", inf]", sep='')
            print("\nThe t-statistic equals", t)
            print("The p-value equals", pvalue)
        if(t < -k):
            if show: print("\nTherefore, H0", colored("is rejected.", 'magenta'))
            if count: return 1
            else: return t
        else:
            if show: print("\nTherefore, H0", colored("cannot be rejected.", 'magenta'))
            if count: return 0
            else: return t


def t2samplesMU(X1, X2, simvar = True, show = True, count = False):
    if simvar:
        n = len(X1)
        m = len(X2)
        s21 = np.var(X1)
        s22 = np.var(X2)
        sp = np.sqrt(((n-1)*s21 + (m-1)*s22)/(n+m-2))
        diff = np.abs((np.mean(X1)-np.mean(X2)))
        df = n+m-2

        K = [12.076, 4.303, 3.182, 2.276, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]
        for x in np.linspace(2.201, 1.980, 50):
            K.append(x)
        if df <= 60:
            k = K[df-1]
        else: k = 1.960

        c = k*(sp*np.sqrt(1/n + 1/m))

        if show:
            print("We test", colored("H0: u1-u2 >= 0", 'magenta'),
                "against", colored("H1: u1-u2 < 0", 'magenta'))
            print("\nReject H0 if the difference of means is more than", c)
            print("\nThe difference in means equals", diff)
        if(diff > c):
            if show: print("Therefore, H0", colored("is rejected.", 'magenta'))
            if count: return 1
        else:
            if show: print("Therefore, H0", colored("cannot be rejected.", 'magenta'))
            if count: return 0














