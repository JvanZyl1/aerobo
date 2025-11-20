import botorch
import torch
import math
import numpy as np
from botorch.utils.transforms import unnormalize, normalize

def check_inputs(x):
    # otherwise unnormalised x's not within bounds
    assert x.min()>=0.0 and x.max()<=1.0

class pressure_vessel():
    '''
    Pressure Vessel Problem
    '''
    def __init__(
            self,
            dtype,
            device
            ):
        
        from botorch.test_functions import PressureVessel
        self.fun = PressureVessel(negate=True).to(dtype=dtype, device=device)
        
        self.best_value = 5868.764836
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        self.bounds = self.fun.bounds
        self.num_constraints = self.fun.num_constraints
        self.name = 'pressurevessel'

    def objective(self,x):
        t = 0.6224 * x[0] * x[2] * x[3] \
                + 1.7781 * x[1] * x[2]**2 \
                + 3.1661 * x[0]**2 * x[3] \
                + 19.84 * x[0]**2 * x[2]
        return torch.tensor(t)

    def constraints(self,x,num_add_constraints=100):
        c1 = -x[0] + 0.0193*x[2]
        c2 = -x[1] + 0.00954*x[2]
        c3 = -torch.pi * x[2]**2 * x[3] - (4/3) * torch.pi * x[2]**3 + 1296000
        c4 = x[3] - 240 
        c = (c1, c2, c3, c4) 
        return torch.tensor(c)

    def eval_objective(self,x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        check_inputs(x)
        return -self.objective(unnormalize(torch.tensor(x), self.fun.bounds))

    def eval_constraints(self,x):
        check_inputs(x)
        return self.constraints(unnormalize(torch.tensor(x), self.fun.bounds))

class speedreducer_model():
    '''
    SPEEDREDUCER PROBLEM
    '''
    def __init__(
            self,
            dtype,
            device
            ):
        
        from botorch.test_functions import SpeedReducer
        self.fun = SpeedReducer(negate=True).to(dtype=dtype, device=device)
        
        self.best_value = 2996.3482
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        self.bounds = self.fun.bounds
        self.num_constraints = self.fun.num_constraints
        self.name = 'speedreducer'

    def objective(self,x):
        t =  0.7854*x[0]*x[1]**2 * (3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
                - 1.508*x[0] * (x[5]**2 + x[6]**2) \
                + 7.4777 * (x[5]**3 + x[6]**3) \
                + 0.7854 * (x[3]*x[5]**2 + x[4]*x[6]**2)
        return torch.tensor(t)

    def constraints(self,x):
        c1 = 27*x[0]**(-1)*x[1]**(-2)*x[2]**(-1) - 1
        c2 = 397.5 * x[0]**(-1) *x[1]**(-2) *x[2]**(-2) - 1
        c3 = 1.93*x[1]**(-1) *x[2]**(-1) *x[3]**(3) *x[5]**(-4) - 1
        c4 = 1.93*x[1]**(-1) *x[2]**(-1) *x[4]**(3) *x[6]**(-4) - 1
        c5 = (1)/(0.1*x[5]**3) * (((745*x[3])/(x[1]*x[2]))**2+16.9e6)**(0.5) - 1100
        c6 = (1)/(0.1*x[6]**3) * (((745*x[4])/(x[1]*x[2]))**2+157.5e6)**(0.5) - 850
        c7 = x[1]*x[2] - 40
        c8 = (x[0]/x[1]) * (-1) + 5
        c9 = (x[0]/x[1]) - 12
        c10 = (1.5*x[5] + 1.9) * x[3]**(-1) - 1
        c11 = (1.1*x[6] + 1.9) * x[4]**(-1) - 1
        c = (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) 
        return torch.tensor(c)

    def eval_objective(self,x):
        check_inputs(x)
        """This is a helper function we use to unnormalize and evalaute a point"""
        return -self.objective(unnormalize(torch.tensor(x), self.fun.bounds))

    def eval_constraints(self,x):
        check_inputs(x)
        return self.constraints(unnormalize(torch.tensor(x), self.fun.bounds))

class ackley():

    def __init__(
        self,
        dtype,
        device
        ):
    
        from botorch.test_functions import Ackley
        
        self.dim = 10
        self.best_value = 0
        self.bounds = torch.ones((2,10))
        self.bounds[0,:], self.bounds[1,:] = self.bounds[0,:] * (-5), self.bounds[0,:]* (10)
        self.lb, self.ub = self.bounds
        self.num_constraints = 2
        self.name = 'ackley'

    def objective(self,x):
        
        f = -20 * torch.exp(-0.2*torch.sqrt((1/self.dim)*torch.sum(x**2))) \
            - torch.exp((1/self.dim)*torch.sum(torch.cos(2*torch.pi*x))) \
                + 20 + torch.exp(torch.tensor(1))
        
        return f

    def eval_objective(self,x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return -self.objective(unnormalize(torch.tensor(x), self.bounds))

    def constraints(self,x):

        g1 = x.sum()
        g2 = torch.norm(x, p=2) - 5
        g = (g1,g2)

        return torch.tensor(g)

    def eval_constraints(self,x):
        return self.constraints(unnormalize(torch.tensor(x), self.bounds))

class keane():

    def __init__(
        self,
        dtype,
        device           
        ):

        self.dim = 30
        bounds = torch.ones(2,self.dim).to(dtype=dtype, device=device)
        bounds[0,:]  = 0
        bounds[1,:]  = 10
        self.bounds = bounds
        self.num_constraints = 2
        self.name = 'keane'
        self.best_value = -0.65
        self.lb, self.ub = self.bounds

        # suggested
        # num_constraints = 2
        # batch_size = 50
        # n_init = 100    
        # num_iter = 2000

    def objective(self,x):

        t1 = (torch.cos(x)**4).sum()
        t2 = (torch.cos(x)**2).prod() * 2
        d = x.shape[0]
        weights = torch.arange(1,d+1)
        t3 = torch.sqrt((weights * x**2).sum())
        return -torch.tensor(-torch.abs((t1-t2)/t3))

    def eval_objective(self,x):
        return self.objective(unnormalize(x, self.bounds))
        
    def constraints(self,x):
        c1 = 0.75 - x.prod()
        c2 = x.sum() - 7.5*self.dim

        c = (c1, c2)

        return torch.tensor(c)
    
    def eval_constraints(self,x):
        return self.constraints(unnormalize(x, self.bounds))
    
class weldedBeam():

    def __init__(
        self,
        dtype,
        device           
        ):
        from botorch.test_functions import WeldedBeamSO
        self.fun = WeldedBeamSO().to(dtype=dtype, device=device)
        self.best_value = 2.38119
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        self.num_constraints = 4
        self.name = 'weldedbeam'

    def objective(self, x):
        x1, x2, x3, x4 = x.unbind(-1)
        return 1.10471 * x1.pow(2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    
    def eval_objective(self,x):
        check_inputs(x)
        return -self.objective(unnormalize(torch.tensor(x), self.fun.bounds))
    
    def constraints(self,x):
        x1, x2, x3, x4 = x.unbind(-1)
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        t_max = 13000.0
        s_max = 30000.0
        d_max = 0.25

        M = P * (L + x2 / 2)
        R = torch.sqrt(0.25 * (x2.pow(2) + (x1 + x3).pow(2)))
        J = 2 * math.sqrt(2) * x1 * x2 * (x2.pow(2) / 12 + 0.25 * (x1 + x3).pow(2))
        P_c = (
            4.013
            * E
            * x3
            * x4.pow(3)
            * 6
            / (L**2)
            * (1 - 0.25 * x3 * math.sqrt(E / G) / L)
        )
        t1 = P / (math.sqrt(2) * x1 * x2)
        t2 = M * R / J
        t = torch.sqrt(t1.pow(2) + t1 * t2 * x2 / R + t2.pow(2))
        s = 6 * P * L / (x4 * x3.pow(2))
        d = 4 * P * L**3 / (E * x3.pow(3) * x4)

        g1 = t - t_max
        g2 = s - s_max
        g3 = x1 - x4
        #g4 = 0.10471 * x1.pow(2) + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0
        g5 = d - d_max
        g6 = P - P_c

        return torch.stack([g1, g2, g3, g5, g6], dim=-1)
        
    def eval_constraints(self,x):
        check_inputs(x)    
        return torch.tensor(self.constraints(unnormalize(torch.tensor(x), self.fun.bounds)))
    
class tension_compression_string():

    def __init__(
            self,
            dtype,
            device
            ):
        
        from botorch.test_functions import TensionCompressionString
        self.fun = TensionCompressionString().to(dtype=dtype, device=device)
        
        self.dim = self.fun.dim
        self.lb, self.ub = self.fun.bounds
        self.bounds = self.fun.bounds
        self.num_constraints = self.fun.num_constraints
        self.name = 'tensincompressionstring'
        self.best_value = 0

    def objective(self, X):
        x1, x2, x3 = X.unbind(-1)
        return x1.pow(2) * x2 * (x3 + 2)

    def constraints(self, X):
        x1, x2, x3 = X.unbind(-1)
        constraints = torch.stack(
            [
                1 - (x2.pow(3) * x3) / (71785 * x1.pow(4)),
                (4 * x2.pow(2) - x1 * x2) / (12566 * x1.pow(3) * (x2 - x1))+ 1 / (5108 * x1.pow(2))- 1,
                1 - 140.45 * x1 / (x3 * x2.pow(2)),
                (x1 + x2) / 1.5 - 1,
            ],
            dim=-1,
        )
        return constraints

    def eval_objective(self,x):
        check_inputs(x)
        return -self.objective(unnormalize(torch.tensor(x), self.fun.bounds))
    
    def eval_constraints(self,x):
        # here we need a minus because output of the slack function 
        # is already multiplied by -1
        check_inputs(x)
        return torch.tensor(self.constraints(unnormalize(torch.tensor(x), self.fun.bounds)))
    

class rover():

    def __init__(self, dtype, device):
        from benchmark_rover.rover_function import Rover
        self.fun = Rover()
        
        self.dim = self.fun.dims
        self.lb, self.ub = self.fun.lb,self.fun.lb
        self.bounds = self.fun.bounds
        self.num_constraints = self.fun.num_constraints
        self.name = 'rover'
        self.best_value = 0

    def objective(self, x):
        f,_ = self.fun(x)
        return f

    def constraints(self,x):
        _,c = self.fun(x)
        return c

    def eval_objective(self,x):
        check_inputs(x)
        xx = np.array(unnormalize(torch.tensor(x), self.fun.bounds))
        return torch.tensor(self.objective(xx))
    
    def eval_constraints(self,x):
        check_inputs(x)
        xx = np.array(unnormalize(torch.tensor(x), self.fun.bounds))
        return torch.tensor(self.constraints(xx))

    

