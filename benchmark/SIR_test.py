
# coding: utf-8

# In[47]:


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

from pathlib import Path
Path.ls = lambda p: [i for i in p.iterdir()]


def fullDisplay(df,max_rows=None,max_col=None,width=None):
    df_cp = df.style.set_properties( **{'width': f'{width}px'}) if width is not None else df.copy()
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_col,):
        display(df_cp)

def pickleDump(obj,fpath):
    with open(fpath,'wb') as f:
        pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def pickleLoad(fpath):
    with open(fpath,'rb') as f:
        return pickle.load(f)

    
target = pd.read_csv('target_counties.csv')
Pop = target[['pop']].values.flatten()
County = target[['county']].values.flatten()
State = target[['state']].values.flatten()


# In[48]:


US_death_path = Path('./../US/time_series_us_deaths_NYT.csv')
US_death_df = pd.read_csv(US_death_path);
dates = US_death_df.columns.to_list()



US_dfs = {'death':US_death_df, 'dates':dates}

# group by country

country_dfs = {}
for k,df in US_dfs.items():
#     df=df.drop(['Lat','Long'],axis=1)
#     df.groupby('state').agg(sum)
    country_dfs[k]=df


# In[49]:


from matplotlib.collections import LineCollection

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N # 
    dIdt = beta * S * I / N - gamma * I 
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def SIR(beta,gamma,nt,N,I0,R0):
    """
    N - Total population
    I0, R0 = 1, 0 Initial number of infected and recovered individuals, I0 and R0.
    S0 = N - I0 - R0 Everyone else, S0, is susceptible to infection initially.
    beta, gamma - Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    """
    t = np.linspace(0, nt, nt)
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - 10*I0 - 1000*R0
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S,I,R


def fit(I, D, N, loss,n_ittr=5,
            n_train=None,n_test=None,
            beta_range=(0.1,1.5),gamma_range=(1e-5, 0.5),
            case_weight=1):
    
    n_total = n_train+n_test
    # train-test split
        
    I_train=I[-n_total:-n_test]
    I_test=I[-n_test:]
    D_train=D[-n_total:-n_test]
    D_test=D[-n_test:]

    #ODE init vals
    I0,R0 = I_train[0],D_train[0]
    S = N - I - 1000*D

    # creat param pairs for grid search
    params = []
    for beta in np.linspace(*beta_range,n_ittr):
        for gamma in np.linspace(*gamma_range,n_ittr):
#                 for case_weight in np.logspace(-10, -6, 10):
            for case_weight in [0]:
                params.append((beta,gamma,I_train,D_train,case_weight,N))

    # run grid search
    pool = multiprocessing.Pool(processes=4)
    mse_arr = pool.map(sim_sir_paral,params)
    pool.close()
    pool.join() 

    beta,gamma = params[np.argmin(mse_arr)][:2]
    return beta,gamma


def predict(params, IDR, N, n_predict=None):
    beta, gamma = params
#         I,D,R = self._I,self._D,self._R
    I,D = IDR[0],IDR[1]
    I0,D0 = I[0],D[0]
    s,i,d = SIR(beta, gamma, n_predict, N, I0, D0)
    
    return s,i,d
    
    
def sim_sir_paral(params):
    betta,gamma,I,D,case_weight,N = params
    I0, D0 = I[0],D[0]

    s,i,d = SIR(betta,gamma, len(I), N, I0, D0)

    if(case_weight):
        mse = np.sum((case_weight*(I-i)**2) + (D-d)**2)
        logI = np.log(I+1)
        logi = np.log(i+1)
        logD = np.log(D+1)
        logd = np.log(d+1)

    else:
#             logdD = np.where(np.log(abs(D-d))<=0, 0, np.log(np.abs(D-d)))
        logD = np.log(D+1)
        logd = np.log(d+1)
        if (loss == 'mse'):
            mse = np.mean((I-i)**2) 
        elif (loss == 'log'):
#             mse = np.sum((logD - logd)**2)
            mse = np.sum((np.log(I+1)-np.log(i+1))**2)
        else:
            sigma_square = (D*(N-D)/N)
            sigma_square = np.where(sigma_square>0,sigma_square,1)
            mse = np.sum(((d-D)**2)/sigma_square)

    return mse


# In[50]:


def run(loss,county, state, pop, n_train,n_test):
    
    betas = []
    gammas = []
    
    countydata = usCounties[(usCounties.county==county)&(usCounties.state==state)].sort_values(by='date')
    
    # get data
    I = countydata.cases.values
    D = countydata.deaths.values
#         N = Pop[index]
    beta, gamma = fit(I, D, pop, loss,case_weight=0, n_train = n_train, n_test=n_test)
    nt = n_train+n_test
#         s, i, d = data.predict([I[n_train:], D[n_train:], R[n_train:]], n_predict = nt-n_train)
    s, i, d = predict((beta,gamma),[I, D],pop,n_predict=nt)
    i_train,i_test=i[-nt:-n_test],i[-n_test:]
    return (i_train,i_test,I)
    
    


# In[53]:


n_trains=[7, 14, 28]
n_tests=[7,14] 

Loss = ['mse']
usCounties = pd.read_csv('./../benchmark/us-counties.csv')
pop = Pop[0]
county = County[0]
state = State[0]

metric = pd.DataFrame()
method=[]
train_size=[]
test_size=[]
regions = []
rel_err = []
abs_err = []
for pop, county, state in zip(Pop, County, State):
    for loss in Loss:
        for n_train in n_trains:
            for n_test in n_tests:
                train_size.append(n_train)
                test_size.append(n_test)
                method.append(f'SIR_{loss}')
                regions.append(county)
                i_train,i_test,I = (run(loss, county, state.strip(), pop, n_train=n_train,n_test=n_test))
                rel_err.append(abs(i_test[-1]-I[-1])/I[-1])
                abs_err.append(abs(i_test[-1]-I[-1]))
                
metric['method']=method
metric['train_size']=train_size
metric['test_size']=test_size
metric['county']=regions
metric['rel_err']=rel_err
metric['abs_err']=abs_err
            
            
            


# In[55]:


metric.to_csv('SIR_metric.csv')


# In[ ]:




