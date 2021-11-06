from skopt import gp_minimize
from skopt.space import Integer
import random
random.seed(1234)

##unknown true objective function
def objective(param_list):
    val=(param_list[0]-9)**1 + (param_list[1]-11)**2
    print ("x="+str(param_list[0])+" y="+str(param_list[1])+" objective="+str(val))
    return val 


def start_bo_engine():
     ##define the search space
     space = [
        Integer(name="x", low=-25, high=25),
        Integer(name="y", low=-25, high=25),
    ]
     ##call the BO
     res=gp_minimize(objective,
                     space,
                     n_calls=10,
                     n_random_starts=5,
                     random_state=1234)

start_bo_engine()
