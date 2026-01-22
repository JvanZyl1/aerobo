# Adapted from BoTorch (https://github.com/meta-pytorch/botorch), licensed under the MIT License.

import gpytorch
import torch
from botorch.utils.transforms import unnormalize, normalize
from dataclasses import dataclass
#from aerobo import *
from aerobo_v1 import *
import benchmarks as bmp
from benchmark_extension import *
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = True


def run_aerobo(
        M, # numerical model to optimise
        ld_in, # number latent input dimensions 
        ld_out, # number of latent output dimensions 
        num_iter, # maximum number of iteration 
        batch_size, # batch size for BO 
        lr,
        num_epochs_init,
        num_epochs,
        n_init, # initial samples 
        n_experiments, # number of total experiments
        benchmark_name,
        path,
        dtype, # choose the data type
        device, # choose the device cpu/gpu
        file_name: str = None,
        input_reduction: bool = True,
        acqf: str = 'cts',
        turbo: bool = True,
        updating: bool = True,
        joint_learning: bool = True,
        weighted_training: bool = False, 
        lookback: int = None,  
        ):
    
    # array were data is stored
    data = []
    
    # START FOR LOOP TO PERFORM N EXPERIMENTS
    scbo = aerobo_scbo(M)

    for experiment in range(n_experiments):
        
        sobol = SobolEngine(M.dim, scramble=True, seed=0)

        # Define example state (ScboState is dataclass )
        state = ScboState(dim=ld_in, batch_size=batch_size,num_constraints=ld_out)
        print(state)
        N_CANDIDATES = 2000 #min(5000, max(2000, 200 * M.dim)) if not SMOKE_TEST else 4

        train_X, train_Y, train_C = compute_doe(model=M, n_init=n_init)        

        model=AEROBO_variationalGP(            
            input_dim=train_X.shape[1],
            output_dim=train_C.shape[1],
            ld_in=ld_in,
            ld_out=ld_out,
            input_reduction=input_reduction,
            weighted_training=weighted_training,
            dtype=dtype
            )

        # init training    
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            if joint_learning:
                # update model with new data
                model.joint_learning(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_C=train_C,
                    num_epochs=num_epochs_init,
                    lr=lr,
                    lookback=None # here set to None because we want to train on the whole initial data set
                    )
            elif not joint_learning:
                model.non_joint_learning(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_C=train_C,
                    num_epochs=num_epochs_init,
                    lr=lr,
                    lookback=None # here set to None because we want to train on the whole initial data set
                    )
        
        # ---------------------- 
        # ----START BO LOOP-----
        # ----------------------
        while len(train_X) < num_iter:
            
            # Generate a batch of candidates
            X_next = scbo.generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=train_C,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                turbo=turbo,
                acqf=acqf,
                sobol=sobol
            )

            # Evaluate both the objective and constraints for the selected candidaates
            Y_next = torch.tensor([], dtype=dtype, device=device)
            C_next = torch.tensor([], dtype=dtype, device=device)
            for x in X_next:
                y = M.eval_objective(x)
                c = M.eval_constraints(x)
                Y_next      = torch.cat((Y_next,y.unsqueeze(-1)),dim=0)
                C_next      = torch.cat((C_next,c.unsqueeze(-1)),dim=1)
            Y_next = Y_next.unsqueeze(-1)
            C_next = C_next.T

            # Update TuRBO state
            state = scbo.update_state(state=state, Y_next=Y_next, C_next=C_next)

            # Append data. Note that we append all data, even points that violate
            # the constraints. This is so our constraint models can learn more
            # about the constraint functions and gain confidence in where violations occur.
            train_X     = torch.cat((train_X, X_next), dim=0)
            train_Y     = torch.cat((train_Y, Y_next), dim=0)
            train_C     = torch.cat((train_C, C_next), dim=0)

            # Print current status. Note that state.best_value is always the best
            # objective value found so far which meets the constraints, or in the case
            # that no points have been found yet which meet the constraints, it is the
            # objective value of the point with the minimum constraint violation.
            if (state.best_constraint_values <= 0).all():
                print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
            else:
                violation = state.best_constraint_values.clamp(min=0).sum()
                print(
                    f"{len(train_X)}) No feasible point yet! Smallest total violation: "
                    f"{violation:.2e}, TR length: {state.length:.2e}"
                )
            
            # following 2024Maus - JoCo
            lookback = max(min(lookback,train_X.shape[0]),batch_size) if lookback else None

            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                if updating and joint_learning:
                    # update model with new data
                    model.joint_learning(
                        train_X=train_X,
                        train_Y=train_Y,
                        train_C=train_C,
                        num_epochs=num_epochs,
                        lr=lr,
                        lookback=lookback
                        )
                elif updating and not joint_learning:
                    model.non_joint_learning(
                        train_X=train_X,
                        train_Y=train_Y,
                        train_C=train_C,
                        num_epochs=num_epochs,
                        lr=lr,
                        lookback=lookback
                        )

            data_exp = {
                'train_X': train_X, 
                'train_Y': train_Y,
                'train_C': train_C,
            }
            
        data.append(data_exp)
            
        if file_name is None:
            torch.save(data,'{}/{}_aerobo.pt'.format(path,benchmark_name))
        else: 
            torch.save(data,'{}/{}_aerobo_{}.pt'.format(path,benchmark_name,file_name))


if __name__ == '__main__':

    n_experiments   = 5
    max_cholesky_size = float("inf")  # Always use Cholesky

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}

    path = os.path.dirname(os.path.realpath(__file__))        
    
    benchmark_list = [
        #'speedreducer',
        #'pressure_vessel',
        #'weldedbeam',
        #'tensioncompressionstring',
        'rover',
        #'keane'
        ]

    for benchmark_name in benchmark_list:

        # select benchmark based on benchmar_name
        if benchmark_name == 'speedreducer':
            M = bmp.speedreducer_model(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 100, 1, 20
        elif benchmark_name == 'pressure_vessel':
            M = bmp.pressure_vessel(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 200, 3, 20
        elif benchmark_name == 'ackley':
            M = bmp.ackley(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 200, 3, 20
        elif benchmark_name == 'weldedbeam':
            M = bmp.weldedBeam(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 200, 3, 20
        elif benchmark_name == 'tensioncompressionstring':
            M = bmp.tension_compression_string(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 200, 3, 20
        elif benchmark_name == 'rover':
            M = bmp.rover(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 2500, 100, 100
        elif benchmark_name == 'keane':
            M = bmp.keane(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 2000, 50, 100
        elif benchmark_name == 'mopta':
            from benchmark_mopta import *
            M = mopta(dtype=dtype, device=device)
            num_iter, batch_size, n_init = 2000, 10, 130
        else: print('Choose valid benchmark!')

        # # EXTEND THE EXISTING BENCHMARK
        # M = extend_benchmark_with_random_constraints(
        #     M=M,
        #     D=200,
        #     G=500,
        #     c_latent=2,
        #     nl_subspace=True
        #     )

        # input_dim = [10,20,60]

        # for ld in input_dim:
            
        #     run_aerobo(
        #         M                   =M,
        #         ld_in               =ld,
        #         ld_out              =10,
        #         num_iter            =num_iter,
        #         batch_size          =batch_size,
        #         lr                  =0.1,
        #         num_epochs_init     =20,
        #         num_epochs          =20,
        #         n_init              =n_init,
        #         n_experiments       =20,
        #         benchmark_name      =M.name,
        #         path                =path,
        #         dtype               =dtype,
        #         device              =device,
        #         file_name           =f'ld_in_{ld}',
        #         input_reduction     =True, 
        #         acqf                ='cts',
        #         turbo               =True,
        #         updating            =True,
        #         joint_learning      =True,
        #         weighted_training   =True,
        #         lookback            =None
        #     )
        
        run_aerobo(
            M                   =M,
            ld_in               =15,
            ld_out              =1,
            num_iter            =num_iter,
            batch_size          =batch_size,
            lr                  =0.1,
            num_epochs_init     =20,
            num_epochs          =20,
            n_init              =n_init,
            n_experiments       =n_experiments,
            benchmark_name      =M.name,
            path                =path,
            dtype               =dtype,
            device              =device,
            file_name           =f'ld_in',
            input_reduction     =True, 
            acqf                ='cts',
            turbo               =True,
            updating            =True,
            joint_learning      =True,
            weighted_training   =True,
            lookback            =None
            )