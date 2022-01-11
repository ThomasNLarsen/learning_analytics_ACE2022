import numpy as np
import do_mpc
from casadi import *
import matplotlib.pyplot as plt

S, C, K = 1, 1, 9

T_min = 10
T_max = 100

### ELEMENTS OF MODEL DYNAMICS ###
def decay(x, T, T_min, T_max):
    '''
    Decay the knowledge in each skill wrt. the size of the timestep, T.
    '''
    alpha = 0.001  # HYPERPARAMETER: decay rate
    return np.exp(-alpha*T) * x  # np.exp(alpha*T) * x
    #return 0.9*x


def potential_improvement(x, w, h):
    '''
    Calculate the maximum improvement possible based on the difference between required skill vs. current skill,
    for the skills involved in the task.
    '''
    # TODO: THIS FACTOR HAS A VERY NARROW BAND
    beta = 0.1  # HYPERPARAMETER: minimal improvement --- |h - x| = 1 --- beta*epsilon/(beta + 1) = constant
    epsilon = 0.1  # HYPERPARAMETER: potential improvement --- |h - x| = 0 --- beta*epsilon/(beta + 0) = epsilon
    #return w * (beta * epsilon) / (beta + norm_1(h - x))
    return w * (beta * epsilon) / (beta + fabs(h - x))


def prerequisite_deficiencies(x, w):
    '''
    Discount the knowledge improvement for the involved skills by identifying potential deficits in prerequisite skills.
    '''
    #_P = np.array(
    #    [
    #        [0, 1, 1, 0],
    #        [0, 0, 1, 0],
    #        [0, 0, 0, 0],
    #        [0, 0, 0, 0]
    #    ]
    #)
    '''
        Describes the prerequisites structure as an Acyclic Directed Graph (DAG), or a binary tree
        We have:
            9 skills total, for instance
                                            Skill 1
                                           /       \
                                    Skill 2         Skill 3
                                   /      \        /       \
                            Skill 4   Skill 5   Skill 6    Skill 7
                           /                                      \
                       Skill 8                                  Skill 9

        For which, Skill 1 is prerequisite to Skill 2 and Skill 3, and so on.
        Prerequisites are inherited.
    '''
    # Cell [i,j] describes whether skill i is prerequisite for skill j
    _P = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    '''
        x*I^T           = 4x4 matrix, [
                                       [x_1, ..., x_4],
                                       ...,
                                       [x_1, ..., x_4]
                                      ]
        x*I^T \cdot P = 4x4 matrix,   [
                                       [P_11*x_1, ..., P_14*x_4],
                                       ...,
                                       [P_41*x_1, ..., P_44*x_4]
                                      ]
        w \cdot (x*I^T \cdot P) =     [
                                       [w_1*P_11*x_1, ..., w_1*P_14*x_4],
                                       ...,
                                       [w_4*P_41*x_1, ..., w_4*P_44*x_4]
                                      ]
        horzsplit --> split into list of row vectors
        Now multiply the elements in each list of row vectors together:
                                      [
                                       (w_1*P_11*x_1) * ... * (w_1*P_14*x_4),
                                       ...,
                                       (w_4*P_41*x_1) * ... * (w_4*P_44*x_4)
                                      ]
        But only multiply the non-zero entries.      
    '''
    p_matrix = w * (x @ SX.ones(1,K) * _P)
    p_vectors = horzsplit(p_matrix)
    p_factors = SX.ones(K,1)
    for i in range(K):
        for j in range(K):
            # Multiply non-zero entries
            p_factors[i] *= if_else(p_vectors[i][j] > 0, p_vectors[i][j], 1)
            # TODO: Bug - if initial x is 0.0, then p_vectors[i,j] is zero and the prerequisite factor becomes 1,
            # TODO: even if it shouldn't be.
            # Hotfix: initialize x0 as a small value, e.g., 1e-4.
    #print(repr(p_factors))
    return p_factors


def complements(x, w):
    '''
        Describes the complementary structure as a ladder overlaying the Acyclic Directed Graph (DAG)
        We have, as before:
            9 skills total,
                                            Skill 1
                                           /       \
                                    Skill 2 ------- Skill 3
                                   /      \        /       \
                            Skill 4   Skill 5 -- Skill 6   Skill 7
                           /                                      \
                       Skill 8  ------------------------------  Skill 9

        For which, the horizontal "ladder steps" represents the skills that are complementary to each other.
    '''
    gamma = 50  # HYPERPARAMETER: bonus learning from involving complementary skills
                # --- smaller gamma -> bigger factor
                # --- bigger gamma  -> smaller factor
    #_C = np.array(
    #    [
    #        [0, 0, 0, 0],
    #        [0, 0, 0, 1],
    #        [0, 1, 0, 1],
    #        [0, 1, 1, 0]
    #    ]
    #)
    _C = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0]
        ]
    )
    n_C = _C.sum(axis=0)  # Get the number of complementing skills for each skill [0, 2, 1, 2]

    '''
        Want:
        [
         (1 + (w_1*C_11*x_1) / (gamma*n_C)) * ... * (1 + (w_4*C_14*x_1) / (gamma*n_C)),
         ...,
         (1 + (w_1*C_41*x_1) / (gamma*n_C)) * ... * (1 + (w_1*C_44*x_1) / (gamma*n_C)),
        ]
        i.e., same procedure as for the prerequisites, but divide row vectors by gamma*n_C[i] and add one before product
    '''

    c_matrix = w * (x @ SX.ones(1, K) * _C)
    c_vectors = horzsplit(c_matrix)
    c_factors = SX.ones(K, 1)
    for i in range(K):
        for j in range(K):
            c_factors[i] *= if_else(c_vectors[i][j] > 0,
                                    1 + c_vectors[i][j] / (gamma*n_C[i]),
                                    1)

    #print(repr(c_factors))

    return c_factors


def time_factor(T, T_min, T_max):
    omega = 0.05  # HYPERPARAMETER: T = T_max --> 1 + omega multiplicative improvement
    return 1 + (omega * (T - T_min) / (T_max - T_min))
##################################


def performance(x, h, w):
    _y = 1 - norm_1(fmax(h - x, 0))/norm_1(w)  # + noise

    return _y


def _rhs(x, u):
    '''
    x: knowledge level of student s --- 4x1 vector
    u: TLA containing [w, h, T]
        w: skill involvement vector --- 4x1
        h: skill taxonomy vector    --- 4x1
        T: time spent on this TLA   --- 1x1
    '''
    T_min = 10
    T_max = 100

    # Unpack model input to skill involvements (w), skill taxonomies (h), and time (T)
    h, T = u
    w = if_else(h > 0.05, 1, 0)
    #x_next = decay(x, T, T_min, T_max) \
    x_next = decay(x, T, T_min, T_max) \
             + potential_improvement(x, w, h) \
             * prerequisite_deficiencies(x, w) \
             * complements(x, w) \
             * time_factor(T, T_min, T_max)

    # TODO: Clip knowledge between 0 and 1 with casadi or not?
    x_next = if_else(x_next > 1, 1, x_next)

    return x_next

def dynamics_model(S, C, K):
# Obtain an instance of the do-mpc model class
    # and select time discretization:
    model_type = 'discrete' # either 'discrete' or 'continuous'
    symvar_type = 'SX'  # Faster matrix multiplication properties than SX
    print("dynamics_model(): model type is", model_type, "and symvars are", symvar_type)
    model = do_mpc.model.Model(model_type, symvar_type=symvar_type)

    # Introduce new states, inputs and other variables to the model, e.g.:
    x = model.set_variable(var_type='_x', var_name='x', shape=(K,1))
    #w = model.set_variable(var_type='_z', var_name='w', shape=(K,1))
    h = model.set_variable(var_type='_u', var_name='h', shape=(K,1))
    T = model.set_variable(var_type='_u', var_name='T')
    T_total = model.set_variable(var_type='_x', var_name='T_total')

    # Set right-hand-side of ODE for all introduced states (_x).
    # Names are inherited from the state definition.
    x_next = _rhs(x, [h, T])  # expected params: x, u -- u = [w, h, T] (4x1, 4x1, 1)
    model.set_rhs('x', x_next)
    model.set_rhs('T_total', T_total + T)
    #model.set_rhs('w', if_else(h > 0, 1, 0))


    # Task complexity limit: 3 skills per task:
    #model.set_alg(expr_name='complexity_limit', expr=sum1(w))

    # State and input monitoring expressions: (and for cost functions later)
    w = model.set_expression(expr_name='w', expr=if_else(h > 0, 1, 0))
    model.set_expression(expr_name='cost_x', expr=sum1(fabs(x)))
    model.set_expression(expr_name='cost_w', expr=fmax(0, sum1(w) - 3))
    #model.set_expression(expr_name='alpha', expr=fmax(1 - x/h, 0) * h)
    #model.set_expression(expr_name='delta_x', expr=fmax(gaussian(fmax(1 - x/h, 0) * h, (1-h)/2 * perturbate(), (1-x)/5), 0))

    # Monitoring expressions
    model.set_expression(expr_name='decay', expr=x - decay(x, T, T_min, T_max))
    model.set_expression(expr_name='pot_impr', expr=potential_improvement(x, w, h))
    model.set_expression(expr_name='prereq', expr=prerequisite_deficiencies(x, w))
    model.set_expression(expr_name='compl', expr=complements(x, w))
    model.set_expression(expr_name='time_factor', expr=time_factor(T, T_min, T_max))
    model.set_expression(expr_name='performance', expr=performance(x, h, w))

    # Time-limit

    # Measurements
    u_meas = model.set_meas('u_meas', h, meas_noise=False)
    #y_meas = model.set_meas('y', performance(x, h, w), meas_noise=False)

    # State measurements (just performance)
    alpha = fmax(h - x,0)
    model.set_expression(expr_name='y_meas', expr=sum1(1 - alpha) / K)
    y_meas = model.set_meas('y_meas', sum1(1 - alpha) / K, meas_noise=True)

    # Setup model:
    model.setup()

    return model



# OLD (from numpy version)
def P(w):
    ''' FUNCTION:
        Map w -> prerequisite skills in P (idxs vector) --> to find relevant skills in x

        Ex:
        w       = [1,   1,   0,   0]
        x       = [0.5, 0.5, 0.5, 0.5]
        factors = [1,   0.5, 1,   1]    --> i.e., skill 2 has skill 1 as prerequisite,
                                                  in which the student is only 0.5 proficient in.
                                                  The other skills are unaffected
        |
        Thus, this function should return the following list of lists:
        |
        map F(w, P) --> [[], [1], [], []], i.e., prerequisite skills for each skill involved in the TLA
        return
    '''
    _P = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
    )

    # Zero out irrelevant columns in P
    _P = _P * w
    print(repr(_P))

    # Get indexes of prerequisite skills in P, for each relevant skill in w.
    p_idxs = []
    for col in range(K):
        p_idxs.append(np.argwhere(_P.T[col] > 0))

    # Returns: the indexes for the skills that are prerequisite for the skills needed in this TLA
    print(p_idxs)
    return p_idxs


# OLD (from numpy version)
def C(w):
    '''
    FUNCTION:
    Map w --> complementary skills in C (idxs vector) --> to find complementary skills in x
    Ex:
    w = [1,   1,   0,   0]
    x = [0.5, 0.5, 0.5, 0.5]
    factors = []
    1: no complementary skills          --> factor = 1
    2: skills 3 and 4 are complementary --> factor = (1 + x_3/gamma*n_compl) * (1 + x_4/gamma*n_compl)
                                                   = (1 + 0.5/10*2) * (1 + 0.5 / 10*2)
                                                   = 1.050625
    3: not involved in TLA              --> factor = 1
    4: not involved in TLA              --> factor = 1
    |
    |
    Thus, this function should return the following list of indexes:
    map F(w, C) --> [[], [2, 3], [], []], i.e., complementary skills for each skill involved in the TLA
    '''
    # From paper: column i describes what skills are complementary to skill i.
    # Here, column 3 describes that skill 4 is complementary to skill 3.
    _C = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]
        ]
    )

    # Zero out irrelevant skills:
    _C = _C * w

    # Get indexes of complementary skills in C, for each relevant skill in w.
    c_idxs = []
    for col in _C.T:
        c_idxs.append(np.argwhere(col > 0))

    # Returns: the indexes for the skills that are prerequisite for the skills needed in this TLA
    return c_idxs

def P_DAG(w):
    '''
        Describes the prerequisites structure as an Acyclic Directed Graph (DAG), or a binary tree
        We have:
            9 skills total, for instance
                                            Skill 1
                                           /       \
                                    Skill 2         Skill 3
                                   /      \        /       \
                            Skill 4   Skill 5   Skill 6    Skill 7
                           /                                      \
                       Skill 8                                  Skill 9

        For which, Skill 1 is prerequisite to Skill 2 and Skill 3, and so on.
            (Or the other way around? -- Skill 2 and 3 are prerequisite to Skill 1)
    '''
    return

def C_DAG(w):
    '''
        Describes the complementary structure as a ladder overlaying the Acyclic Directed Graph (DAG)
        We have, as before:
            9 skills total,
                                            Skill 1
                                           /       \
                                    Skill 2 ------- Skill 3
                                   /      \        /       \
                            Skill 4   Skill 5 -- Skill 6   Skill 7
                           /                                      \
                       Skill 8  ------------------------------  Skill 9

        For which, the horizontal "ladder steps" represents the skills that are complementary to each other.
    '''
    return


def random_TLA():
    _w = np.random.randint(0, 2, size=9)
    _h = np.random.uniform(size=9) * _w
    _T = np.array([10 * np.random.randint(1, 11)])

    #u = np.concatenate((_w, _h, _T))
    u = np.concatenate((_h, _T))
    return u[:, None]


def monotonically_increasing_TLA(x):
    _w = np.ones(K)
    #_h = (x.squeeze() + 0.1*(np.random.random() - 0.5)) * _w
    _h = x.squeeze() * _w
    _h += abs(0.1 * (np.random.random()))
    _h = np.clip(_h, 0, 1)
    _T = np.array([55]) # * np.random.randint(1, 11)
    #u = np.concatenate((_w, _h, _T))
    u = np.concatenate((_h, _T))
    return u[:, None]


def constant_TLA():
    #_w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    #_h = 0.1*np.ones(4) * _w
    #_h = np.array([0.2, 0.4, 0.6, 0.8])
    _h = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    _T = np.array([55])
    #_T = 10*np.ones(4)
    #return np.array([_w, _h]), _T
    #return np.array([_w, _h, _T])
    #u = np.concatenate((_w, _h, _T))
    u = np.concatenate((_h, _T))
    return u[:, None]


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("#### HYPERPARAMETERS ####")
    ############################# decay(x, T, T_min, T_max):
    alpha = 0.001               # Decay rate --> affects all skills
    ############################# potential_improvement(x, a):
    beta = 0.1                  # minimum improvement
    epsilon = 0.1               # maximum (potential) improvement
    ############################# complements(x, w):
    gamma = 50                  # inverse proportional scaling factor per complementary skill
    ############################# time_factor(T, T_min, T_max):
    omega = 0.05                # scaling factor for spending more than the minimum amount of time per TLA.


    print("{:15} = {:0.3f} - {:35}".format("alpha",   alpha,   "knowledge decay"))
    print("{:15} = {:0.2f} - {:35}".format("beta",    beta,    "minimum improvement"))
    print("{:15} = {:0.2f} - {:35}".format("epsilon", epsilon, "maximum (potential) improvement"))
    print("{:15} = {:0.2f} - {:35}".format("gamma",   gamma,   "complement scaling"))
    print("{:15} = {:0.2f} - {:35}".format("omega",   omega,   "time bonus scaling"))
    print("#########################\n")

    print("####### PARAMETERS #######")
    #x0 = np.zeros(K)           # Student knowledge per skill
    init_skills = 0.01*np.ones(K)  # Student knowledge per skill
    init_time = np.array([0.0])
    x0 = np.concatenate((init_skills, init_time))
    print("x0 =", x0)
    #w = np.array([1, 1, 1, 1])           # Skill involvents for TLA
    #print("w0 =", w)
    h = np.array([0.05, 0.05, 0.05, 0.05])   # Skill taxonomies for TLA
    print("h0 =", h)
    T = 10                               # MODEL INPUT 2: Time to spent on the TLA
    T_max = 100
    T_min = 10
    print("T, T_max, T_min =", T, T_max, T_min)
    print("##########################\n")



    print("\nINIT MODEL ... ", end="")
    model = dynamics_model(1, 1, K)
    print("SUCCESS\n")

     # Initialize simulator
    sim = do_mpc.simulator.Simulator(model)
    sim.set_param(t_step=1)
    sim.setup()
    sim.x0 = x0
    sim_graphics = do_mpc.graphics.Graphics(sim.data)

    # Run the simulator
    for i in range(50):
        skills = x0[:K]
        time = x0[-1]
        u0 = monotonically_increasing_TLA(skills)
        #u0 = constant_TLA()
        #u0 = random_TLA()
        x0 = sim.make_step(u0)

        #w0 = u0[:K]
        #h0 = u0[K:-2]
        #T0 = u0[-1]
        h0 = u0[:K]
        T0 = u0[-1]


    print("after 100 iterations:\n", sim.data['_x'][-1])


    # Plotting simulation data
    fig, ax, graphics = do_mpc.graphics.default_plot(sim.data, figsize=(9, 10))
    #sim_graphics.plot_results()
    graphics.plot_results()
    #ax[0].set_ylim(0.0, 1.1)
    plt.show()
    exit()

    # Plotting simualtion data -- offset to differentiate between each skill.
    _x = sim.data['_x'][:, :-1]
    offset = [0.1*i for i in range(K)]
    #offset.append(0)
    _x += offset
    _t = np.ones((50, K))
    #labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'Time Cumulative']
    labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    for i in range(50):
        _t[i, :] += i
    plt.plot(_t, _x, label=labels)
    plt.title("Skill knowledges, labeled and offset from each other for visual clarity")
    plt.legend(loc="lower right")
    plt.show()

    # Plotting input data
    _u = sim.data['_u'][:, :-1]
    #offset = [0.1 * i for i in range(K)]
    # offset.append(0)
    #_u += offset
    _t = np.ones((50, K))
    labels = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9']
    for i in range(50):
        _t[i, :] += i
    plt.plot(_t, _u, label=labels)
    plt.title("Inputs")
    plt.legend(loc="lower right")
    plt.show()

    # Plotting time input
    _u = sim.data['_u'][:, -1]
    # offset = [0.1 * i for i in range(K)]
    # offset.append(0)
    # _u += offset
    _t = np.ones((50, 1))
    labels = ['T']
    for i in range(50):
        _t[i, :] += i
    plt.plot(_t, _u, label=labels)
    plt.title("Inputs, time")
    plt.legend(loc="lower right")
    plt.show()
    exit()
