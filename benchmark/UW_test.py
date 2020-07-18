
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import OrderedDict
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.special import erf
from numpy.random import multivariate_normal
from scipy.special import erf
# multi-processing function
import multiprocessing


target = pd.read_csv('target_counties.csv')
Pop = target[['pop']].values.flatten()
County = target[['county']].values.flatten()
State = target[['state']].values.flatten()


# In[4]:


from scipy.stats import beta

def train_test_split(x, ratio):
    n_train = round(len(x)*ratio)
    x_train = x[0:n_train]
    x_test = x[n_train:]
    return (x_train, x_test)

def integrand(x):
    return np.exp(-x**2)

def ERF(params):
    alpha, beta, p, t = params
    d = (p/2) * (1 + erf(alpha*(t-beta)))
    return d

def ERF2(params):
    alpha, beta, P, t = params
    d = (P/2) * (1 + (2/np.sqrt(np.pi))* quad(integrand, 0, alpha*(t-beta))[0])
    return d


def MSE(params, *args):
    alpha, beta, P = params[0], params[1], params[2]
    ts, Ds, N, loss = args # Ds: True death
    alphas = np.ones(len(ts))*alpha
    betas = np.ones(len(ts))*beta
    Ps = np.ones(len(ts))*P
    params = np.stack((alphas, betas, Ps, ts), axis=-1)
    ds = np.apply_along_axis(ERF, 1, params)
#     ratio1 =  (Ds[-1] - Ds[-2])/(ds[-1] - ds[-2])
#     ratio2 = (ds[-1] - ds[-2])/(Ds[-1] - Ds[-2])
#     factor = (np.log(Ds[-1])-np.log(ds[-1]))**2
    if(loss == 'weight'):
        weights = np.ones(len(Ds))
        weights[-3:]=100
        mse = np.mean((np.log(Ds+1)-np.log(ds+1))**2*Ds**2)
    else:
        mse = np.mean((np.log(Ds+1)-np.log(ds+1))**2)
    return mse


def minimization(params, args):
    # params is alphas, betas, Ps, t
    
    result = minimize(MSE, params, args=args, method = 'L-BFGS-B', bounds = ((0,1), (1, 1e3), (1e0, 1e10)))
    collect = result.x
    success = result.success
    alpha, beta, p = collect[0], collect[1], collect[2]
    ts, Ds, N, loss = args
    alphas = np.ones(len(ts))*alpha
    betas = np.ones(len(ts))*beta
    Ps = np.ones(len(ts))*p
    params = np.stack((alphas, betas, Ps, ts), axis=-1)
    ds = np.apply_along_axis(ERF, 1, params)
    return (ds, alpha, beta, p, success)



def Jacobian(params):
    alpha, beta, p, t = params
    dfda = (p/np.sqrt(np.pi))*np.exp(-alpha**2*(t-beta)**2)*(t-beta)

    dfdb = -(p/np.sqrt(np.pi))*alpha*np.exp(-alpha**2*(t-beta)**2)
    dfdp = ERF(params)/p
    J = np.array([dfda, dfdb, dfdp, 0]).reshape(-1,1)
    return J

# loghessian calculate the hessian of loglik at each time point t
def loghessian(params):
    # diff is the sum of difference between Ds and ds
    alpha, beta, p, t, diff = params
    
    d = ERF((alpha, beta, p, t))
    dfda, dfdb, dfdp, _ = Jacobian((alpha, beta, p, t)).flatten()


    
    J = np.array([dfda, dfdb, dfdp]).reshape(-1,1)
    matrix = J@J.T/d**2
    
    return matrix


def uncertainty(params):
    # t1 is the interpolating series, t2 is the extrapolating series
    alpha, beta, p, t1, t2, sigma2, diff = params
    
    T = len(t1)
    alpha1 = np.ones(len(t1))*alpha
    beta1 = np.ones(len(t1))*beta
    p1 = np.ones(len(t1))*p
    diff1 = diff[0:len(t1)]

    params1 = np.stack((alpha1, beta1, p1, t1, diff1), axis=-1)
    
    
    hess1 = np.sum(np.apply_along_axis(loghessian, 1, params1), axis = 0)/sigma2
    hessian = hess1
    
    
    alpha2 = np.ones(len(t2))*alpha
    beta2 = np.ones(len(t2))*beta
    p2 = np.ones(len(t2))*p
    
    params2 = np.stack((alpha2, beta2, p2, t2), axis=-1)

    d2 = np.apply_along_axis(ERF, 1, params2)

    covariance = inv(hessian)
       
    return covariance


def bootstrap(params):
    D, ts, alpha, beta, p, loss = params
    packs = []
    
    print(alpha, beta, p)
    for i in range(200):
#         Diff = np.concatenate(([0],np.log(D[1:])-np.log(D[:-1])))
#         Diff = beta.pdf((ts+1)/(ts[-1]*1.1), 0.5, 0.5)
#         Diff = np.concatenate(([1], (D[1:] - D[:-1])/D[:-1]))
#         Diff = np.log(D)
        propor = (D[:-1]/np.sum(D[:-1]))
        if loss == 'weight':
            prop = D[len(D)-6:-1]/np.sum(D[len(D)-6:-1])
            t_sample = np.sort(np.random.choice(np.arange(len(D)-6,len(D)-1), 3,p=prop,replace=False))
            t_sample = np.concatenate([t_sample, [len(D)-1]])
        else:
            prop = D[len(D)-6:-1]/np.sum(D[len(D)-6:-1])
            t_sample = np.sort(np.random.choice(np.arange(len(D)-6,len(D)-1), 3,p=prop,replace=False))
            t_sample = np.concatenate([t_sample, [len(D)-1]])
#             t_sample = np.sort(np.random.choice(len(D), int((len(D))*0.6), replace=False))
#             t_sample = np.concatenate([t_sample, [len(D)-1]])
        d_sample = D[t_sample]
        a0 = uniform(alpha*0.1, min(alpha*2,0.1))
        b0 = uniform(0.8*beta, beta*1.2)
        p0 = uniform(0.8*p, 1.2*p)
        _, alpha_hat, beta_hat, p_hat, success = minimization((a0, b0, p0), (t_sample, d_sample, len(d_sample), loss))
        packs.append([alpha_hat, beta_hat, p_hat])
        
    packs = np.array(packs)
    return packs


# In[16]:


def upper_lower(packs, D_sim, ts, CI = 0.95, debug=False):
    Ds = []
    likelis = []
    invalid = 0
    for alpha,beta,p in packs:

        if(alpha <0 or alpha >= 1 or beta <0 or p <0):
            invalid += 1
            print(invalid)
            continue
        if(debug): print(alpha, beta, p)
        D = ERF((alpha, beta, p, ts))


        likelis.append(np.mean((np.log(D_sim+1) - np.log(D+1))**2))

        Ds.append(D)
    Sample = len(Ds)
    print(Sample)

    order = sorted(range(len(likelis)), key=lambda k: likelis[k])
    
    Ds = [Ds[i] for i in order][0:round(Sample*CI)]
    Ds_mat = np.array(Ds)
    D_up =  np.max(Ds_mat,axis=0)
    D_but =  np.min(Ds_mat,axis=0)
    
#     if D_but[0] <= 1:
#         print(D_but)
#         print(packs[Ds_mat[:, 0] <= 1])
#         print('likeli')
#         print(np.mean((np.log(D_but+1) - np.log(D_sim+1))**2))
    
    return (D_up, D_but)


def upper_lower_cases(packs, D_sim, ts, dt, n_train, CI = 0.95, debug=False):
    Ds = []
    likelis = []
    invalid = 0
    for alpha,beta,p in packs:

        if(alpha <0 or alpha >= 1 or beta <0 or p <0):
            invalid += 1
            print(invalid)
            continue
        if(debug): print(alpha, beta, p)
        D = ERF((alpha, beta, p, ts))


        likelis.append(np.mean((np.log(D_sim+1) - np.log(D+1))**2))

        Ds.append(D)
    Sample = len(Ds)
    print(Sample)
    order = sorted(range(len(likelis)), key=lambda k: likelis[k])
    
    Ds = [Ds[i] for i in order][0:round(Sample*CI)]
    Ds_mat = np.array(Ds)
    D_up =  np.max(Ds_mat[:, 1:]-Ds_mat[:, :-1],axis=0)
#     D_up[n_train-1] = max(np.max(Ds_mat[:, n_train]) - dt, np.max(Ds_mat[:, n_train-1])
    D_but =  np.min(Ds_mat[:, 1:]-Ds_mat[:, :-1],axis=0)
#     D_but[n_train] = np.min(Ds_mat[:, n_train-1]) - np.min(Ds_mat[:, n_train-2])
    

    return (D_up, D_but)


# In[19]:


def firstInfected(cum_Infect, N, wdate=False):
    i =  np.argmax(cum_Infect>N/np.exp(14))
    if(wdate): return i,self.dates[i]
    return i
    
    

    
def runERF(loss,county,state,N,n_train,n_test):
    

    countydata = usCounties[(usCounties.county==county)&(usCounties.state==state)].sort_values(by='date')
    
    n_total = n_train+n_test
    I = countydata.cases.values
    
    I = I[-n_total:]
    
    flength = n_total
    nt = n_total
    
    ts = np.linspace(0, nt-1, nt)
    I_train, I_test = I[-n_total:-n_test], I[-n_test:]
    
    t = np.linspace(0, flength-1, flength)
    n_train = len(I) - n_test
    
    
    p0 = 10
    a0 = 0.1
    b0 = 10
    _, alpha, beta, p, success = minimization((a0, b0, p0), (ts[0:n_train], I_train, N, loss))
        

    d = ERF((alpha, beta, p, t))
    return (d[-n_total:-n_test],d[-n_test:],I)


# In[21]:


# state= 'California'
# county = 'Los Angeles'

Lfuncs = ['weight','MLE']


criter = 'cases'
# measure = 'test'
country = 'US'

usCounties = pd.read_csv('./../benchmark/us-counties.csv')

metric = pd.DataFrame()
method=[]
train_size=[]
test_size=[]
regions = []
rel_err = []
abs_err = []

n_trains=[7, 14, 28]
n_tests=[7,14] 

pop, county, state = Pop[0],County[0],State[0]
for loss in Lfuncs:
    for n_train in n_trains:
        for n_test in n_tests:
#     for measure in ['predictions', 'test']:
#     for pop, county, state in zip(Pop, County, State):
            train_size.append(n_train)
            test_size.append(n_test)
            method.append(f'UW_{loss}')
            regions.append(county)
            i_train,i_test, I = runERF(loss, county, state.strip(), pop, n_train=n_train,n_test=n_test)
            rel_err.append(abs(i_test[-1]-I[-1])/I[-1])
            abs_err.append(abs(i_test[-1]-I[-1]))

        
metric['method']=method
metric['train_size']=train_size
metric['test_size']=test_size
metric['county']=regions
metric['rel_err']=rel_err
metric['abs_err']=abs_err
        
        


# In[22]:


metric


# In[ ]:




