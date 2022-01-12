import do_mpc
import numpy as np


# TODO: Not implemented
# Configure the estimator (MHE / EKF / State-feedback)
def mhe_estimator(model):
    # Obtain an instance of the do-mpc MHE class
    # and initiate it with the model.
    # Optionally pass a list of parameters to be estimated.
    K = model._x['x'].shape[0]
    mhe = do_mpc.estimator.MHE(model)

    # Set parameters:
    setup_mhe = {
        'n_horizon': 5,
        't_step': 1,
        'store_full_solution': True,
        'meas_from_data': True,
    }
    mhe.set_param(**setup_mhe)

    P_v = 0.2*np.diag(np.array([1]))      # No measurement output Noise # Confidence about the measure
    P_x = 0.25*np.eye(K+2)                     # State weight
    # P_x[-1][-1] = 0
    # P_x[-2][-2] = 
    #P_p = 10*np.eye(1)                   # No Parameter weight
    P_w = np.eye(K)                 # Process Noise (both on x and y) # Confidence about the process
    # P_w[-1][-1] = 0
    # P_w[-2][-2] = 0

    mhe.set_default_objective(P_x, None, None, P_w)

    # Measurement function:
    y_template = mhe.get_y_template()

    def y_fun(t_now):
        n_steps = min(mhe.data._y.shape[0], mhe.n_horizon)
        for k in range(-n_steps,0):
            y_template['y_meas',k] = mhe.data._y[k]
            # y_template['y_meas',k,'h_meas'] = mhe.data._y[k,:K]
            # y_template['y_meas',k,'T_meas'] = mhe.data._y[k,K]
            # y_template['y_meas',k,'y'] = mhe.data._y[k,K+1]
            # y_template['y_meas',k,'days_meas'] = mhe.data._y[K+2]
            # y_template['y_meas',k,'T_total_meas'] = mhe.data._y[K+3]
        return y_template

    mhe.set_y_fun(y_fun)


     # Lower bounds on states:
    mhe.bounds['lower', '_x', 'x'] = 0
    mhe.bounds['lower', '_x', 'T_total'] = 0

    # Upper bounds on states
    mhe.bounds['upper', '_x', 'x'] = 1
    mhe.bounds['upper', '_x', 'T_total'] = 800

    # Lower bounds on inputs:
    # mpc.bounds['lower', '_u', 'w'] = 0
    mhe.bounds['lower', '_u', 'h'] = 0
    mhe.bounds['lower', '_u', 'T'] = 0.5

    # Upper bounds on inputs:
    # mpc.bounds['upper', '_u', 'w'] = 1
    mhe.bounds['upper', '_u', 'h'] = 1
    mhe.bounds['upper', '_u', 'T'] = 3

    mhe.bounds['upper', '_z', 'scheduling'] = 0

    # [Optional] Set measurement function.
    # Measurements are read from data object by default.

    mhe.setup()

    return mhe