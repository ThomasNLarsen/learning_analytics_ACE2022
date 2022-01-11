import do_mpc
import numpy as np

# Configure the MPC controller
def mpc_controller(model):
    # Obtain an instance of the do-mpc MPC class
    # and initiate it with the model:
    mpc = do_mpc.controller.MPC(model)

    # Set parameters:
    setup_mpc = {
        'n_horizon': 7,
        't_step': 1,
        'n_robust': 0,
        'state_discretization': 'discrete',
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Configure objective function:
    lterm = -model.aux['cost_x']
    mterm = -model.aux['cost_x']
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # mpc.set_rterm(F=0.1, Q_dot = 1e-3) # Scaling for quad. cost, but we don't want to penalize higher difficulties..

    # Lower bounds on states:
    mpc.bounds['lower', '_x', 'x'] = 0
    mpc.bounds['lower', '_x', 'T_total'] = 0

    # Upper bounds on states
    mpc.bounds['upper', '_x', 'x'] = 1
    mpc.bounds['upper', '_x', 'T_total'] = 800

    # Lower bounds on inputs:
    # mpc.bounds['lower', '_u', 'w'] = 0
    mpc.bounds['lower', '_u', 'h'] = 0
    mpc.bounds['lower', '_u', 'T'] = 0.5

    # Upper bounds on inputs:
    # mpc.bounds['upper', '_u', 'w'] = 1
    mpc.bounds['upper', '_u', 'h'] = 1
    mpc.bounds['upper', '_u', 'T'] = 3

    mpc.bounds['upper', '_z', 'scheduling'] = 0
    #mpc.bounds['lower', '_z', 'scheduling'] = -1

    #mpc.bounds['upper', '_z', 'n_involved_skills'] = 3
    # Terminal bounds
    # mpc.terminal_bounds['lower', 'T_total'] = 12
    # mpc.terminal_bounds['upper', 'T_total'] = 12

    #mpc.set_nl_cons('max_skill',  model.aux['skills_involved'], ub=3.1, soft_constraint=True)


    mpc.setup()

    return mpc