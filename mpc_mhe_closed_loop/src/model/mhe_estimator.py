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
        'meas_from_data': True,
    }
    mhe.set_param(**setup_mhe)

    P_v = 0.2*np.diag(np.array([1]))   # No measurement output Noise # Confidence about the measure
    P_x = np.eye(K)                 # State weight
    #P_p = 10*np.eye(1)               # No Parameter weight
    P_w = 0.5*np.eye(K)                 # Process Noise (both on x and y) # Confidence about the process

    mhe.set_default_objective(P_x, P_v, None, P_w)

    # Measurement function:
    y_template = mhe.get_y_template()

    def y_fun(t_now):
        n_steps = min(mhe.data._y.shape[0], mhe.n_horizon)
        for k in range(-n_steps,0):
            y_template['y_meas',k] = mhe.data._y[k]
        return y_template

    mhe.set_y_fun(y_fun)


    # Set bounds for states, parameters, etc.
    mhe.bounds['lower','_x', 'x'] = 0.0
    mhe.bounds['upper','_x', 'x'] = 1.0
    #mhe.bounds['lower','_x', 'y'] = 0
    #mhe.bounds['upper','_x', 'y'] = 1
    mhe.bounds['lower','_u', 'u'] = 0.0
    mhe.bounds['upper','_u', 'u'] = 1.0
    # mhe.bounds['lower','_z', 'z'] = 0
    # mhe.bounds['upper','_z', 'z'] = 0.2

    # [Optional] Set measurement function.
    # Measurements are read from data object by default.

    mhe.setup()

    return mhe