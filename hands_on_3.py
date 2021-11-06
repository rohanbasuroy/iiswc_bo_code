#multiple BOs, kripke
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct,ConstantKernel,WhiteKernel)
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import sklearn.gaussian_process as gp
import subprocess as sp
import shlex
import random
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import statistics
import time 
import os 
FNULL = open(os.devnull, 'w')

#define parameters
max_calls=2000
num_initial_sample=1
percentage_sampled_by_acq=0.5
delay_min=20 #(delay_min - delay_window) must be greater than the (number of models + num_initial_smaple)
delay_max=50
delay_window=2
lookahead_max=10
lookahead_window=4
throttling_times=2.5


param_list=[]
exe_list=[]
n_list=["DGZ","DZG","GDZ","GZD","ZDG","ZGD"]
n_choice=[0,1,2,3,4,5]
g_choice=[1,2,4,8,16,32]
d_choice=[8,16,24,32,48,64,96]
t_choice=[1,2,3,4,6,8,10,12,14,16,18,20]
param_choice=[n_choice, g_choice, d_choice, t_choice]


def hyperthreading(i):
    if i==0: #off
        j=10
        while j <20:
            comm="sudo bash -c 'echo 0 > /sys/devices/system/cpu/cpu"+str(j)+"/online'"
            os.system(comm)
            j+=1
    if i==1: #on
        j=10
        while j<20:
            comm="sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu"+str(j)+"/online'"
            os.system(comm)
            j+=1

def objective(list):
    delay_min=20 ##########
    n=n_list[int(list[0])]
    g=int(list[1])
    d=int(list[2])
    t=int(list[3])    
    command="OMP_NUM_THREADS="+str(t)+" "+"./kripke.exe --groups 64 --quad 192 --layout "+n+" "+"--gset "+str(g)+" "+"--dset "+str(d)
    start_time=time.time()
    os.system(command)
    end_time=time.time()
    exe_time=end_time-start_time
    print ("the execution time is", exe_time)
    param_list.append([n,g,d,t])
    exe_list.append(exe_time)
    return -1.0*exe_time
    
    #print (n,g,d,t,u)
    #r=(random.randint(1,10))
    #exe_list.append(r)
    #return (r)

def surrogate(model, XX):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(XX, return_std=True)
def acquisition_ei(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    Z=((mu - best) / (std+1E-9))
    #print (mu - best)* norm.cdf(Z) + std*norm.pdf(Z)
    return (mu - best)* norm.cdf(Z) + std*norm.pdf(Z)
def acquisition_pi(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
def acquisition_ucb(XX, Xsamples, model):
    yhat, _ = surrogate(model, XX)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:]
    v=1
    delta=0.1
    d=len(param_choice)
    t=len(exe_list)
    Kappa = np.sqrt( v* (2*  np.log( (t**(d/2. + 2))*(np.pi**2)/(3. * delta)  )))
    return mu + Kappa*(std+1E-9)
    
def opt_acquisition(XX, yy, model, acqval):
    total_choice=1
    for cc in range(len(param_choice)):
        total_choice*=len(param_choice[cc])
    Xsamples=[]
    out_count=0
    while out_count < percentage_sampled_by_acq*total_choice:
        in_list=[]
        in_count=0
        while in_count < len(param_choice):
            in_list.append(random.choice(param_choice[in_count]))
            in_count+=1
        if in_list not in XX :
            Xsamples.append(in_list)
        out_count+=1
    if acqval==0:
        scores = acquisition_ei(XX, Xsamples, model)
    if acqval==1:
        scores = acquisition_pi(XX, Xsamples, model)
    if acqval==2:
        scores = acquisition_ucb(XX, Xsamples, model)
    ix = argmax(scores)
    return Xsamples[ix] 

kernel_options=[gp.kernels.DotProduct()+ gp.kernels.WhiteKernel(), gp.kernels.Matern(length_scale=1.0, nu=1.5),
                gp.kernels.RBF(length_scale=1.0),gp.kernels.RationalQuadratic(length_scale=1.0)]
#gp.kernels.ExpSineSquared(length_scale=1.0)
#[gp.kernels.DotProduct() + gp.kernels.WhiteKernel()

#model1-4:EI, model5-8:PI, model9-12:UCB
model1 = GaussianProcessRegressor(kernel=kernel_options[0])
model2 = GaussianProcessRegressor(kernel=kernel_options[1])
model3 = GaussianProcessRegressor(kernel=kernel_options[2])
model4 = GaussianProcessRegressor(kernel=kernel_options[3])
model5 = GaussianProcessRegressor(kernel=kernel_options[0])
model6 = GaussianProcessRegressor(kernel=kernel_options[1])
model7 = GaussianProcessRegressor(kernel=kernel_options[2])
model8 = GaussianProcessRegressor(kernel=kernel_options[3])
model9 = GaussianProcessRegressor(kernel=kernel_options[0])
model10 = GaussianProcessRegressor(kernel=kernel_options[1])
model11 = GaussianProcessRegressor(kernel=kernel_options[2])
model12 = GaussianProcessRegressor(kernel=kernel_options[3])

model_list=[model1,model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]
model_sampling_list=[[] for i in range(0,len(model_list))]


##reset hardware knobs
hyperthreading(1)
u_comm="likwid-setFrequencies -umax 3.0"
sp.check_output(shlex.split(u_comm), stderr=FNULL)
f_comm="likwid-setFrequencies -c S0 -f 2.2"
sp.check_output(shlex.split(f_comm), stderr=FNULL)


#initial sampling 
XX=[]
out_count=0
while out_count < num_initial_sample:
    in_list=[]
    in_count=0
    while in_count < len(param_choice):
        in_list.append(random.choice(param_choice[in_count]))
        in_count+=1
    XX.append(in_list)
    out_count+=1
yy = [objective(xx) for xx in XX]
for m in model_list:
    m.fit(XX, yy)
    

mm=[]
model_selection_list=[]
i=0
while len(exe_list)<max_calls:
    model_min_list=[]
    if i==0:
        for model in model_list:
            if model == model1 or model == model2 or model == model3 or model == model4:
                acqval=0
            elif model == model5 or model == model6 or model == model7 or model == model8:
                acqval=1
            elif model == model9 or model == model10 or model == model11 or model == model12:
                acqval=2
            xx = opt_acquisition(XX, yy, model, acqval)
            actual = objective(xx)
            model_sampling_list[model_list.index(model)].append(-1*actual)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(actual)
            model.fit(XX, yy)
            if model==model1:
                mm.append("m1")
            elif model==model2:
                mm.append("m2")
            elif model==model3:
                mm.append("m3")
            elif model==model4:
                mm.append("m4")
            elif model==model5:
                mm.append("m5")
            elif model==model6:
                mm.append("m6")
            elif model==model7:
                mm.append("m7")
            elif model==model8:
                mm.append("m8")
            elif model==model9:
                mm.append("m9")
            elif model==model10:
                mm.append("m10")
            elif model==model11:
                mm.append("m11")
            elif model==model12:
                mm.append("m12")
            model_selection_list.append(model)
            i+=1
    else:
        
            model=random.choice(model_selection_list)
            if model == model1 or model == model2 or model == model3 or model == model4:
                acqval=0
            elif model == model5 or model == model6 or model == model7 or model == model8:
                acqval=1
            elif model == model9 or model == model10 or model == model11 or model == model12:
                acqval=2
            xx = opt_acquisition(XX, yy, model, acqval)
            actual = objective(xx)
            model_sampling_list[model_list.index(model)].append(-1*actual)
            est, _ = surrogate(model, [xx])
            XX.append(xx)
            yy.append(actual)
            model.fit(XX, yy)
            for m in model_sampling_list:
                model_min_list.append(min(m))
            model_selection_list.append(model_list[model_min_list.index(min(model_min_list))])
            if model==model1:
                mm.append("m1")
            elif model==model2:
                mm.append("m2")
            elif model==model3:
                mm.append("m3")
            elif model==model4:
                mm.append("m4")
            elif model==model5:
                mm.append("m5")
            elif model==model6:
                mm.append("m6")
            elif model==model7:
                mm.append("m7")
            elif model==model8:
                mm.append("m8")
            elif model==model9:
                mm.append("m9")
            elif model==model10:
                mm.append("m10")
            elif model==model11:
                mm.append("m11")
            elif model==model12:
                mm.append("m12")
            i+=1

#with open('param_list.txt', 'w') as f:
#    for item in param_list:
#        print >> f, item

#with open('exe_list.txt', 'w') as f:
#    for item in exe_list:
#        print >> f, item

#with open('model_list.txt', 'w') as f:
#    for item in mm:
#        print >> f, item


##reset hardware knobs
hyperthreading(1)
u_comm="likwid-setFrequencies -umax 3.0"
sp.check_output(shlex.split(u_comm), stderr=FNULL)
f_comm="likwid-setFrequencies -c S0 -f 2.2"
sp.check_output(shlex.split(f_comm), stderr=FNULL)
