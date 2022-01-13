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

#from src.util.plotting import plotter

from casadi import lt


def tla_generation():
    C1 = 0*np.eye(100, 5)
    C1[:,0] = np.random.uniform(low=0.0, high=1.0, size=(len(C1),))
    C1 = C1[C1[:,0].argsort()]
    C2 = 0*np.eye(100, 5)
    C2[:,:2] = np.random.uniform(low=0.0, high=1.0, size=(len(C1),2))
    C2 = C2[C2[:,1].argsort()]
    C3 = 0*np.eye(100, 5)
    C3[:,:3] = np.random.uniform(low=0.0, high=1.0, size=(len(C1),3))
    C3 = C3[C3[:,2].argsort()]
    C4 = 0*np.eye(100, 5)
    C4[:,:4] = np.random.uniform(low=0.0, high=1.0, size=(len(C1),4))
    C4 = C4[C4[:,3].argsort()]
    C5 = 0*np.eye(100, 5)
    C5[:,:5] = np.random.uniform(low=0.0, high=1.0, size=(len(C1),5))
    C5 = C5[C5[:,4].argsort()]
    tla = np.concatenate((C1, C2, C3, C4, C5))
    idx1 = np.sort(np.append(np.random.randint(100, size=4), 99))
    idx2 = np.sort(np.append(np.random.randint(100, size=4), 99))
    idx3 = np.sort(np.append(np.random.randint(100, size=4), 99))
    idx4 = np.sort(np.append(np.random.randint(100, size=4), 99))
    idx5 = np.sort(np.append(np.random.randint(100, size=4), 99))
    scheduled_tla = 0*np.eye(32,5)
    scheduled_tla[:5,:] = C1[idx1,:]
    scheduled_tla[5:10,:] = C2[idx2,:]
    scheduled_tla[10:15,:] = C3[idx3,:]
    scheduled_tla[15:20,:] = C4[idx4,:]
    scheduled_tla[20:25,:] = C5[idx5,:]
    scheduled_tla[25:,:] = C5[np.random.randint(100, size=len(scheduled_tla[25:,:])),:]
    return tla, scheduled_tla


if __name__ == '__main__':
    np.random.seed(seed=42)
    # Obtain all configured modules and run the loop

    phi, psi = data_generation()  # students, questions
    tla, schedule_tla = tla_generation()
    model = dynamics_model(S, C, K)
    mpc = mpc_controller(model)
    mhe = mhe_estimator(model)
    sim = simulator(model)
    #estimator = do_mpc.estimator.StateFeedback(model)

    ''' Use different initial state for the true system (simulator) and for MHE / MPC '''
    x0_true = np.concatenate((phi[4], np.array([0])))

    # MHE/MPC init x
    x0 = np.zeros(model.n_x)
    x0[0] = np.random.uniform(low=0.0, high=0.6)
    x0[1] = np.random.uniform(low=0.0, high=x0[0])
    x0[2] = np.random.uniform(low=0.0, high=x0[0])
    x0[3] = np.random.uniform(low=0.0, high=x0[1])
    x0[4] = np.random.uniform(low=0.0, high=x0[1])

    # Initialize MPC, Sim and MHE
    mpc.x0 = x0
    sim.x0 = x0_true
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
    result_filename = 'results_main_single_student'
    #result_filename = 'results_benchmarkRandom'
    #result_filename = 'results_benchmarkIncreasing'

    q_tracker = np.zeros((S, 2 * C), dtype=bool)
    #q_tracker = 0
    if True:#not os.path.exists(result_path + result_filename + '.pkl'):
        for k in range(50):
            # Find optimal question
            u0_tilde = mpc.make_step(x0)

            _h = u0_tilde#[:K]  # [ideal question difficulties]
            _T = u0_tilde[-1]  # time usage

            # Select the question closest to optimal.
            # Select the question closest to optimal.
            if k % 7 == 0 or k % 7 == 2 or k % 7 == 4:
                _h, q_tracker = question_selector(_h, tla, 0, q_tracker)
                #_h = _h.squeeze()
            else:
                _h = np.zeros(K)

            #u0 = np.concatenate((_h, _T))[:, np.newaxis]
            u0 = _h[:, np.newaxis]

            # Simulate with process and measurement noise
            y_next = sim.make_step(u0,
                                   # v0=0.01 * np.random.randn(model.n_v, 1),
                                   w0=0.01 * np.random.randn(model.n_w, 1))

            # Estimate the next knowledge status
            x0 = mhe.make_step(y_next)


        # Plotting simulation data
        fig, ax, graphics = do_mpc.graphics.default_plot(sim.data, figsize=(9, 10))
        graphics.plot_results()
        plt.show()
        exit()

        do_mpc.data.save_results([sim, mpc, mhe], result_name=result_filename, result_path=result_path, overwrite=True)



