import do_mpc

# Configure the DAE / ODE / Discrete simulator
def simulator(model):
    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    sim = do_mpc.simulator.Simulator(model)

    # Set parameter(s):
    sim.set_param(t_step = 1)

    # Optional: Set function for parameters and time-varying parameters.

    # Setup simulator:
    sim.setup()

    return sim