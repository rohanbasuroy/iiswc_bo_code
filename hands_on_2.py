from skopt import gp_minimize
from skopt.space import Integer, Categorical
import random
random.seed(1234)
import time

##unknown true objective function
def objective(param_list):
    val=(param_list[0]-9)**1 + (param_list[1]-11)**2
    print ("x="+str(param_list[0])+" y="+str(param_list[1])+" objective="+str(val))
    print("Time for iteration: ", time.time()-time_var[len(time_var)-1])
    time_var.append(time.time())
    return val 


def start_bo_engine():
    ##define the search space
    space = list()
    space.append(Categorical([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25], name='x'))
    space.append(Integer(-50, 50, name='y'))
    
     ##call the BO
    res=gp_minimize(objective,
                     space,
                     n_calls=15,
                     n_random_starts=5,
                     random_state=1234)
time_var=[]
time_var.append(time.time())
start_bo_engine()
