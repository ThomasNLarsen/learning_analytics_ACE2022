import do_mpc
import numpy as np
import matplotlib.pyplot as plt

from data.data_generation import data_generation
from src.model.dynamics_model import dynamics_model
from src.model.mpc_controller import mpc_controller
from src.model.mhe_estimator import mhe_estimator
from src.model.simulator import simulator
from src.util.question_selector import question_selector, benchmark1, benchmark2
from src.util.constants import *

from src.util.plotting import plotter

from casadi import lt

# In the case that a dedicated estimator is required, another python file should be added to the project.
# from estimator import estimator

if __name__ == '__main__':
    np.random.seed(seed=42)
    # Obtain all configured modules and run the loop

    phi, psi = data_generation()
    # Manually insert maximum difficulty questions to be sure of reaching 1
    for i in range(1,5):
        psi[-i] = np.ones(K) - 2*i/100
    q_tracker = np.zeros((S,2*C), dtype=bool)
    for i in range(len(phi)):   # for each student i, you need to implement a controller
        model = dynamics_model(S, C, K)
        mpc = mpc_controller(model)
        mhe = mhe_estimator(model)
        sim = simulator(model)
        #estimator = do_mpc.estimator.StateFeedback(model)

        ''' Use different initial state for the true system (simulator) and for MHE / MPC '''
        x0_true = phi[i]
        x0 = np.zeros(model.n_x).reshape((model.n_x, 1))                  # --> Shape = 10

        # Initialize MPC, Sim and MHE
        mpc.x0 = x0
        sim.x0 = x0_true
        mhe.x0 = x0
        # estimator.x0 = x0

        # Set initial guess for MHE/MPC based on initial state.
        mpc.set_initial_guess()
        mhe.set_initial_guess()



        """
        Run MPC main loop:
        """
        import os
        result_path = './src/simulations/mpc_mhe_model/'
        result_filename = f'student_%i'%i
        #result_filename = 'results_benchmarkRandom'
        #result_filename = 'results_benchmarkIncreasing'

        q_previous = x0_true

        if True:#not os.path.exists(result_path + result_filename + '.pkl'):
            for k in range(20):
                # Find optimal question
                u0_tilde = mpc.make_step(x0)

                # Select the question closest to optimal.
                u0, q_tracker = question_selector(u0_tilde, psi, i, q_tracker)

                ## Benchmark 1: Random selection
                #u0 = benchmark1(psi)

                ## Benchmark 2: Monotonically increasing
                #u0 = benchmark2(psi, q_previous)
                q_previous = u0
                # Simulate with process and measurement noise
                x_prev = sim.x0
                y_next = sim.make_step(u0,
                                       v0=0.01 * np.random.randn(model.n_v, 1),
                                       w0=0.05 * np.random.randn(model.n_w, 1))
                # if lt(sim.x0, x_prev):
                #     sim.x0 = x_prev
                # Estimate the next knowledge status
                x0 = mhe.make_step(y_next)


            #save_dict(data_dict)
            do_mpc.data.save_results([sim, mpc, mhe], result_name=result_filename, result_path=result_path, overwrite=True)




    '''
    Plotting the results
    '''
    #fig = plotter(result_path+result_filename+'.pkl', include_mpc_u=True)
    #fig.write_image(result_path+result_filename+'.png')
    # See plotting.py

