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

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
#mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
color = plt.rcParams['axes.prop_cycle'].by_key()['color']


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


def manual_scheduling(scheduled_tla, tracker):
    return scheduled_tla[tracker]


def configure_plot(mpc, mhe, sim):
    mpc_plot = do_mpc.graphics.Graphics(mpc.data)
    mhe_plot = do_mpc.graphics.Graphics(mhe.data)
    sim_plot = do_mpc.graphics.Graphics(sim.data)

    ax[0].set_title('Estimated knowledge:')
    sim_plot.add_line('_x', 'x', ax[0])
    mhe_plot.add_line('_x', 'x', ax[0])

    ax[0].legend(
        sim_plot.result_lines['_x', 'x'] + mhe_plot.result_lines['_x', 'x'],
        ['Recorded', 'Estimated'], title='Knowledge')

    ax[1].set_title('TLA time:')
    #sim_plot.add_line('_x', 'T_total', ax[1])
    #sim_plot.add_line('_u', 'T', ax[1])

    #ax[1].legend(
    #    sim_plot.result_lines['_x', 'T_total'] + sim_plot.result_lines['_u', 'T'],
    #    ['T total', 'TLA duration']
    #)

    ax[2].set_title('TLA difficulty:')
    sim_plot.add_line('_u', 'h', ax[2])
    # mpc_plot.add_line('_u', 'h', ax[2])

    for mpc_line_i, mhe_line_i, sim_line_i in zip(mpc_plot.result_lines.full, mhe_plot.result_lines.full,
                                                  sim_plot.result_lines.full):
        mhe_line_i.set_color(sim_line_i.get_color())
        mhe_line_i.set_color(sim_line_i.get_color())
        mpc_line_i.set_linestyle('--')
        mhe_line_i.set_alpha(0.5)
        mhe_line_i.set_linewidth(5)

    ax[0].set_ylabel('knowledge')
    ax[1].set_ylabel('TLA time [h]')
    ax[2].set_ylabel('TLA difficulty')
    ax[2].set_xlabel('time [day]')

    for ax_i in ax:
        ax_i.axvline(1.0)

    # fig.tight_layout()
    # plt.ion()

if __name__ == '__main__':
    np.random.seed(seed=0)

    '''
    Setup graphic
    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os

    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    # mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    fig.align_ylabels()


    '''
    Set initial state
    '''
    phi, psi = data_generation()  # phi: students, psi: questions (not used)
    tla, schedule_tla = tla_generation()

    #q_tracker = np.zeros((S,2*C), dtype=bool)
    q_tracker = np.zeros((S, len(tla)), dtype=bool)
    for i in range(S):   # for each student i, you need to implement a controller
        model = dynamics_model(S, C, K)
        mpc = mpc_controller(model)
        mhe = mhe_estimator(model)
        sim = simulator(model)
        # estimator = do_mpc.estimator.StateFeedback(model)

        configure_plot(mpc, mhe, sim)

        ''' Use different initial state for the true system (simulator) and for MHE / MPC '''
        # True / simulator init x
        #if os.path.exists('./src/simulations/mpc_mhe_model/average_x0.npy'):
        #    x0_true = np.load('./src/simulations/mpc_mhe_model/average_x0.npy')
        #else:
        #    x0_true = np.append(np.mean(phi, axis=0), np.array([0]))
        #    np.save('./src/simulations/mpc_mhe_model/average_x0.npy', x0_true)

        x0_true = np.concatenate((phi[i], np.array([0])))

        # MHE/MPC init x
        x0 = np.zeros(model.n_x)
        x0[0] = np.random.uniform(low=0.0, high=0.6)
        x0[1] = np.random.uniform(low=0.0, high=x0[0])
        x0[2] = np.random.uniform(low=0.0, high=x0[0])
        x0[3] = np.random.uniform(low=0.0, high=x0[1])
        x0[4] = np.random.uniform(low=0.0, high=x0[1])
        #x0 = np.zeros(model.n_x).reshape((model.n_x, 1))

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
        result_path = './src/simulations/benchmark_schedule/'
        #result_path = './src/simulations/mpc_mhe_model/'
        result_filename = f'student_%i'%i
        #result_filename = 'results_benchmarkRandom'
        #result_filename = 'results_benchmarkIncreasing'
        use_mpc = False
        tracker = 0

        if True:#not os.path.exists(result_path + result_filename + '.pkl'):
            #TLA_sequence = np.load('src/simulations/mpc_mhe_model/average_student_0.pkl', allow_pickle=True)['simulator']['_u']
            for k in range(60):
                if use_mpc:
                    # Find optimal question
                    u0_tilde = mpc.make_step(x0)

                    _h = u0_tilde[:K]  # [ideal question difficulties]
                    #_T = u0_tilde[-1]  # time usage

                    # Select the question closest to optimal.
                    # Select the question closest to optimal.
                    if k % 7 == 0 or k % 7 == 2 or k % 7 == 4:
                        _h, q_tracker = question_selector(_h, tla, i, q_tracker)
                    else:
                        _h = np.zeros(K)

                else:
                    ## Benchmark: Manual schedule
                    #_T = [3]
                    if k % 7 == 0 or k % 7 == 2 or k % 7 == 4:
                        _h = manual_scheduling(schedule_tla, tracker)
                        tracker += 1
                    else:
                        _h = np.zeros(K)

                    ## MPC precalculated average student TLA sequence
                    #_h = TLA_sequence[k]#, :K]
                    #_T = TLA_sequence[k, K:]

                u0 = _h[:, np.newaxis]
                #u0 = np.concatenate((_h, _T))[:, np.newaxis]

                # Simulate with process and measurement noise
                y_next = sim.make_step(u0,
                                       #v0=0.01 * np.random.randn(model.n_v, 1),
                                       w0=0.01 * np.random.randn(model.n_w, 1))

                # Estimate the next knowledge status
                x0 = mhe.make_step(y_next)


            #save_dict(data_dict)
            do_mpc.data.save_results([sim, mpc, mhe], result_name=result_filename, result_path=result_path, overwrite=True)
            #exit()

    '''
    Plotting the results
    '''
    #fig = plotter(result_path+result_filename+'.pkl', include_mpc_u=True)
    #fig.write_image(result_path+result_filename+'.png')
    # See plotting.py

