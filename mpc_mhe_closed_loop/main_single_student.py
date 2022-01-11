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

    phi, psi = data_generation()  # students, questions, involvements
    model = dynamics_model(S, C, K)
    mpc = mpc_controller(model)
    mhe = mhe_estimator(model)
    sim = simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)

    ''' Use different initial state for the true system (simulator) and for MHE / MPC '''
    x0_true = phi[4]
    x0 = np.zeros(model.n_x).reshape((model.n_x, 1))                  # --> Shape = 10

    # Initialize MPC, Sim and MHE
    mpc.x0 = x0
    sim.x0 = np.concatenate((x0_true, np.array([0,0])))
    mhe.x0 = x0
    #estimator.x0 = x0

    # Set initial guess for MHE/MPC based on initial state.
    mpc.set_initial_guess()
    mhe.set_initial_guess()

    """
    Run MPC main loop:
    """
    import os
    result_path = './src/util/'
    result_filename = 'results'
    #result_filename = 'results_benchmarkRandom'
    #result_filename = 'results_benchmarkIncreasing'

    q_previous = x0_true
    q_tracker = np.zeros((S, 2 * C), dtype=bool)

    if True:#not os.path.exists(result_path + result_filename + '.pkl'):
        T_max = 800
        T_cumulative = 0
        for k in range(50):
            # Find optimal question
            u0_tilde = mpc.make_step(x0)

            #_w = u0_tilde[:K]        # [1,   1,   0, 0, 0, 0, 0, 0, 0]
            _h = u0_tilde[:K]     # [0.5, 0.2, 0, 0, 0, 0, 0, 0, 0]
            _T = u0_tilde[-1]        # 50

            # Force max 3 skills involved (not working)
            #_h_threshold = np.sort(_h)[::-1][2]
            #_h[_h < _h_threshold] = 0
            #u0_tilde[:K] = _h

            # Select the question closest to optimal.
            __h, q_tracker = question_selector(_h, psi, 1, q_tracker)
            u0_tilde = np.concatenate((__h, _T))[:, np.newaxis]
            #print("w:", __w)
            #print("h:", __h)
            #print("T:", _T)
            #u0 = np.concatenate((__w, __h, _T), axis=0)[:, np.newaxis]
            #u0 = np.concatenate((__h, _T), axis=0)[:, np.newaxis]

            ## Benchmark 1: Random selection
            #u0 = benchmark1(psi)

            ## Benchmark 2: Monotonically increasing
            #u0 = benchmark2(psi, q_previous)
            q_previous = u0_tilde
            # Simulate with process and measurement noise
            x_prev = sim.x0
            y_next = sim.make_step(u0_tilde,
                                   w0=0.01 * np.random.randn(model.n_w, 1))
                                         #v0=0.01 * np.random.randn(model.n_v, 1),


            # Estimate the next knowledge status
            x0 = mhe.make_step(y_next)
            #x0 = estimator.make_step(y_next)

            # Break if max time exceeded:
            #T_cumulative += _T
            #if T_cumulative >= T_max:
            #    break

        # Plotting simulation data
        fig, ax, graphics = do_mpc.graphics.default_plot(sim.data, figsize=(9, 10))
        # sim_graphics.plot_results()
        graphics.plot_results()
        # ax[0].set_ylim(0.0, 1.1)
        plt.show()

        #save_dict(data_dict)
        do_mpc.data.save_results([sim, mpc, estimator], result_name=result_filename, result_path=result_path, overwrite=True)



    exit()
    '''
    Plotting the results
    '''
    fig = plotter(result_path+result_filename+'.pkl', include_mpc_u=True)
    fig.write_image(result_path+result_filename+'.png')
    # See plotting.py