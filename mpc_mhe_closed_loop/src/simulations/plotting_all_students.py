import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from do_mpc.data import load_results
import numpy as np
import pandas as pd

S = 50
K = 5

#https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
#https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
#CMAP = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']
CMAP = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']  # Wong

def figure_setup(include_mpc_u, nrows=1, ncols=2, sharedx=False, sharedy=False, vspace=0.02):
    fig = make_subplots(rows=nrows, cols=ncols,
                        shared_xaxes=sharedx,
                        shared_yaxes=sharedy,
                        vertical_spacing=vspace
                        )

    fig.update_layout(template='simple_white',
                      font=dict(size=20),
                      width=900,
                      height=1000)

    fig.update_xaxes(
       gridcolor="#EEEEEE",
       showgrid=True,
       linewidth=1,
       mirror=True
    )
    fig.update_yaxes(
       gridcolor="#EEEEEE",
       showgrid=True,
       linewidth=1
    )

    if include_mpc_u:
        fig.layout.update({'yaxis5': dict(
            overlaying='y3',
            side='right',
            anchor='free',
            position=1,
            range=[0, 2],
            tickvals=[0, 1],
            ticktext=[0, 1],
            showgrid=False,
            title=r"$\tilde{u}(k)$"
            )}
        )

    return fig

def extract_data(path):
    df = load_results(path)
    sim_data = df['simulator']
    mpc_data = df['mpc']
    mhe_data = df['estimator']

    # Extract data from do-mpc data structures
    _t = mhe_data['_time']
    x_est = mhe_data['_x', 'x']
    x_true = sim_data['_x', 'x']
    u = mhe_data['_u', 'u']
    u_optimal = mpc_data['_u', 'u']
    y_true = sim_data['_aux', 'y_meas']
    y_est = mhe_data['_aux', 'y_meas']

    return _t, x_est, x_true, u, u_optimal, y_true, y_est

def plotter(path, include_mpc_u=False):
    # Load data:
    _t, x_est, x_true, u, u_optimal, y_true, y_est = extract_data(path)

    _t = _t.flatten()
    n_skills = x_est.shape[1]

    # Initialize axes:
    fig = figure_setup(include_mpc_u, nrows=4, ncols=1, sharedx=True)

    # Knowledge estimation error
    fig.add_trace(go.Scatter(x=_t, y=np.zeros(_t.shape), line=dict(color='#000000'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=_t, y=np.mean(abs(x_est - x_true), axis=1),
                             line=dict(color=CMAP[-2]),
                             name=r"$|\hat{x}(k)-x(k)|$",
                             legendgroup='1',
                             showlegend=True,
                             fill='tonexty'),
                  row=1, col=1)

    for skill in range(n_skills):
        # Estimated knowledge
        fig.add_trace(go.Scatter(x=_t, y=x_est[:, skill], line=dict(color=CMAP[-skill-1]),
                                 name=r"$\theta_{}$".format(skill+1),
                                 legendgroup='1',
                                 #showlegend=False,
                                 showlegend=True), row=1, col=1)

        # True knowledge
        fig.add_trace(go.Scatter(x=_t, y=x_true[:, skill], line=dict(color=CMAP[-skill-1]),
                                 name="Simulated knowledge",
                                 showlegend=False), row=2, col=1)

        # Input
        if include_mpc_u:
            fig.add_trace(go.Scatter(x=_t, y=u[:, skill] + 1, line=dict(shape='hv', color=CMAP[-skill-1]),
                                     name="Question inputs",
                                     showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=_t, y=u_optimal[:, skill], line=dict(shape='hv', color=CMAP[-skill-1]),
                                     name="MPC optimal question",
                                     yaxis='y5',
                                     xaxis='x3',
                                     showlegend=False))#, row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=_t, y=u[:, skill], line=dict(shape='hv', color=CMAP[-skill - 1]),
                                     name="Question inputs",
                                     showlegend=False), row=3, col=1)

    # True and predicted performance
    fig.add_trace(go.Scatter(x=_t, y=y_true.flatten(), line=dict(color=CMAP[0]),
                             name=r"$y(k)$",
                             legendgroup='2',
                             #showlegend=False,
                             showlegend=True), row=4, col=1)
    fig.add_trace(go.Scatter(x=_t, y=y_est.flatten(), line=dict(color=CMAP[1]),
                             name=r"$\hat{y}(k)$",
                             showlegend=True,
                             #showlegend=False,
                             legendgroup='2'), row=4, col=1)

    #
    fig.add_trace(go.Scatter(x=_t, y=np.zeros(_t.shape), line=dict(color=CMAP[-2]), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=_t, y=abs(y_est.flatten() - y_true.flatten()), line=dict(color=CMAP[-2]),
                             name=r"$|\hat{y}(k)-y(k)|$",
                             legendgroup='2',
                             #showlegend=False,
                             showlegend=True,
                             fill='tonexty'), row=4, col=1)

    # Global
    fig.update_xaxes(
        tickvals=_t,
    )
    fig.update_layout(
        xaxis4_title=r"$\text{Question #}$",
        legend={'traceorder': 'grouped'},
        legend_tracegroupgap=500
    )

    # Local subplot changes
    fig.update_yaxes(
        title=r"$\hat{x}(k)$",
        range=[0, 1],
        # tickvals=np.linspace(0,n_skills-1,n_skills),
        # ticktext=[0]*n_skills,
        row=1
    )
    fig.update_yaxes(
        title=r"$x(k)$",
        range=[0, 1],
        # tickvals=np.linspace(0, n_skills - 1, n_skills),
        # ticktext=[0] * n_skills,
        row=2
    )
    if include_mpc_u:
        fig.update_yaxes(
            title=r"$u(k)$",
            range=[0, 2],
            tickvals=[1, 2],
            ticktext=[0, 1],
            secondary_y=False,
            row=3
        )
        fig.update_yaxes(
            showgrid=False,
            secondary_y=True,
            row=3
        )
    else:
       fig.update_yaxes(
            title=r"$u(k)$",
            range=[0, 1],
            row=3
        )
    #fig.update_yaxes(
    #    range=[0, 2],
    #    tickvals=[0, 1],
    #    ticktext=[0, 1],
    #    secondary_y=True,
    #    showgrid=False,
    #    row=3
    #)

    fig.update_yaxes(
        title=r"$\hat{y}(k) \text{ vs. } y(k)$",
        range=[0, 1],
        row=4
    )
    #fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(color='black', size=20)))
    #fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(color='black', size=20)))

    print(fig.layout)
    fig.show()

    return fig


if __name__ == '__main__':
    result_path = './src/simulations/mpc_mhe_model/'
    frames = []
    columns = ['x_est', 'x_true', 'u', 'u_optimal', 'y_true', 'y_est']
    for i in range(0,S):
        result_filename = f'student_%i.pkl'%i
        _t, x_est, x_true, u, u_optimal, y_true, y_est = extract_data(os.path.join(result_path, result_filename))
        #df_tmp = pd.DataFrame(data=np.array([x_est, x_true, u, u_optimal, y_true, y_est]), columns=columns, index=_t.flatten())
        df_tmp = pd.DataFrame({'x_est':[x_est],
                              'x_true':[x_true],
                              'u':[u],
                              'u_optimal':[u_optimal],
                              'y_true':[y_true],
                              'y_est': [y_est]}, index=_t.flatten())
        frames.append(df_tmp)
        print(df_tmp)
        break
    df = pd.concat(frames, axis=0)


    #fig = plotter(os.path.join(result_path, result_filename), include_mpc_u=True)
    #fig.write_image('plots/mpc_mhe_model.png')
    '''
    # Load data:
    _t, x_est, x_true, u, u_optimal, y_true, y_est = extract_data(path_random)

    _t = _t.flatten()
    n_skills = x_est.shape[1]

    # Initialize axes:
    fig = figure_setup(nrows=4, ncols=1, sharedx=True)

    # Knowledge estimation error
    fig.add_trace(go.Scatter(x=_t, y=np.zeros(_t.shape), line=dict(color=CMAP[-2])), row=1, col=1)
    fig.add_trace(go.Scatter(x=_t, y=np.mean(abs(x_est-x_true), axis=1), line=dict(color=CMAP[0]), fill='tonexty'), row=1, col=1)

    for skill in range(n_skills):
        # Estimated knowledge
        fig.add_trace(go.Scatter(x=_t, y=x_est[:, skill], line=dict(color=CMAP[skill])), row=1, col=1)

        # True knowledge
        fig.add_trace(go.Scatter(x=_t, y=x_true[:, skill], line=dict(color=CMAP[skill])), row=2, col=1)

        # Input
        fig.add_trace(go.Scatter(x=_t, y=u[:, skill]+1, line=dict(shape='hv', color=CMAP[skill])), row=3, col=1)
        fig.add_trace(go.Scatter(x=_t, y=u_optimal[:, skill], line=dict(shape='hv', color=CMAP[skill])), row=3, col=1)

    # True and predicted performance
    fig.add_trace(go.Scatter(x=_t, y=y_true.flatten(), line=dict(color='#000000')), row=4, col=1)
    fig.add_trace(go.Scatter(x=_t, y=y_est.flatten(), line=dict(color=CMAP[4])), row=4, col=1)

    #
    fig.add_trace(go.Scatter(x=_t, y=np.zeros(_t.shape), line=dict(color=CMAP[-2])), row=4, col=1)
    fig.add_trace(go.Scatter(x=_t, y=abs(y_est.flatten()-y_true.flatten()), line=dict(color=CMAP[-2]), fill='tonexty'), row=4, col=1)

    fig.update_yaxes(
        title="Estimated knowledge",
        range=[0,1],
        #tickvals=np.linspace(0,n_skills-1,n_skills),
        #ticktext=[0]*n_skills,
        row=1
    )
    fig.update_yaxes(
        title="Simulated knowledge",
        range=[0, 1],
        #tickvals=np.linspace(0, n_skills - 1, n_skills),
        #ticktext=[0] * n_skills,
        row=2
    )
    fig.update_yaxes(
        title="Control inputs",
        range=[0, 2],
        tickvals=[0,1,2],
        ticktext=[0,'0/1',1],
        row=3
    )
    fig.update_yaxes(
        title="Predicted vs. true performance",
        range=[0, 1],
        row=4
    )
    fig.update_xaxes(
        tickvals=_t
    )

    fig.show()
    '''
