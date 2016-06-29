"""Given a probe input of 1, this model outputs whatever values were learned
to sum previously. Given a probe input of 0, none of the neural ensembles fire.

No negative sums, although this is theoretically possible with Nengo
"""

import nengo

# input function constants
t_scale = 1
learn_time = 0.05
t_learn_A = 0.05
t_learn_B = 0.5
p_time = 1.

# configure the neural
pos_only = nengo.Config(nengo.Ensemble)
pos_only[nengo.Ensemble].intercepts = nengo.dists.Uniform(0., 1.)
pos_only[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
pos_only[nengo.Ensemble].eval_points = nengo.dists.Uniform(0., 1.)


def in_func(t):
    #t_step = t % (1 * t_scale)
    t_step = t 
    if t_step > t_learn_A * t_scale and t_step < (t_learn_A + learn_time) * t_scale:
        return 0.1
    elif t_step > t_learn_B * t_scale and t_step < (t_learn_B + learn_time) * t_scale:
        return 0.3
    else:
        return 0.

def probe_func(t):
    #t_step = t % (1 * t_scale)
    t_step = t
    if t_step > t_learn_A * t_scale and t_step < (t_learn_A + learn_time) * t_scale:
        return 1.
    elif t_step > t_learn_B * t_scale and t_step < (t_learn_B + learn_time) * t_scale:
        return 1.
    elif t_step > p_time * t_scale:
        return 1.
    else:
        return -0.1


with nengo.Network() as model:

    in_nd = nengo.Node(in_func)

    probe_nd = nengo.Node(probe_func)

    with pos_only:
        # make this a threshold ensemble
        in_ens = nengo.Ensemble(100, 1)

        # make this a threshold ensemble too
        post_ens = nengo.Ensemble(100, 1)

    nengo.Connection(probe_nd, in_ens)
    learn_conn = nengo.Connection(in_ens, post_ens, learning_rule_type=nengo.PES(0.3e-3), function=lambda x: 0)
    nengo.Connection(in_nd, learn_conn.learning_rule, transform=-1)