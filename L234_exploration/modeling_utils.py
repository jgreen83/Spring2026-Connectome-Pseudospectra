from scipy.optimize import minimize
from scipy import stats
from scipy import linalg
import numpy as np
import functools
import networkx as nx
import tqdm
import collections
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from scipy.optimize import minimize_scalar
from scipy.linalg import expm
from scipy.integrate import simpson
from scipy.sparse.linalg import expm_multiply

import graph_tool.all as gt

def plot_matrix(W: np.ndarray):
    """ Plots a connectivity matrix

    Args:
        W: a weight matrix

    """

    W_extreme_val = np.percentile(np.abs(W[W != 0]), 95) * 1.01
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    im = ax.imshow(W, vmin=-W_extreme_val, vmax=W_extreme_val, cmap="bwr")

    ax.set_ylabel("Postsynaptic neurons", fontsize=12)
    ax.set_xlabel("Presynaptic neurons", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=.5)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Weight (AU)', labelpad=15)

    plt.show()


def plot_synapse_size_distribution(W: np.ndarray, non_zero_only: bool=True):
    """ Plots the distribution of synapse sizes

    Args:
        W: a weight matrix
        non_zero_only: If True, zero weights are not included in the plot

    """


    weights = W.flatten()

    if non_zero_only:
        weights = weights[weights != 0]

    W_extreme_val = np.percentile(np.abs(weights[weights != 0]), 99) * 1.01

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)

    sns.histplot(x=weights, bins=np.linspace(-W_extreme_val, W_extreme_val, 101), ax=ax)
    ax.set_xlabel("Weight (AU)", fontsize=12)
    ax.set_ylabel("Connection count", fontsize=12)
    plt.show()

def input_balance_excitation_inhibition(W: np.ndarray, n_neurons_exc: int, inh_exc_balance: float=1):
    """ Balances E and I on the input side of every neuron

    Args:
        W: a weight matrix
        n_neurons_exc: number of excitatory neurons in the matrix (first n_neurons_exc are assumed to be the excitatory neurons)
        inh_exc_balance: how to balabce E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W_bal: a balanced weight matrix
    """
    W_bal = W.copy()

    per_neuron_exc_input_sum = np.sum(W[:, :n_neurons_exc], axis=1)
    per_neuron_inh_input_sum = np.sum(W[:, n_neurons_exc:], axis=1)

    per_neuron_inh_input_sum[per_neuron_inh_input_sum == 0] = 1 # stable behavior

    inh_scale_up = per_neuron_exc_input_sum / -per_neuron_inh_input_sum * inh_exc_balance
    W_bal[:, n_neurons_exc:] *= inh_scale_up[:, None]
    return W_bal


def generate_random_connectivity_matrix(n_neurons: int, r_inhibitory: float, g: float, connectivity_proba: float,
                                        random_seed: int, balance_W: bool=True, inh_exc_balance: float=1):
    """ Generates random connectivity matrix

    Args:
        n_neurons: number of neurons
        r_inhibitory: ratio of inhibitory neurons
        g: gain
        connectivity_proba: probability of two neurons to be connected (directed)
        random_seed: random seed
        balance_W: whether to balance E and I weights on the input side of every neuron
        inh_exc_balance: how to balabce E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W: a balanced weight matrix
    """

    n_neurons_exc = int(n_neurons * (1 - r_inhibitory))

    # Random Erdos Renyi graph
    er_graph = nx.erdos_renyi_graph(n_neurons, p=connectivity_proba, seed=random_seed, directed=True)
    W_er = nx.to_numpy_array(er_graph)

    # Random log-normally distributed weights
    random_state = np.random.RandomState(random_seed)
    W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons))) * W_er

    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]

    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        # W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
        W = input_balance_quadrants(W, n_neurons_exc, gI=inh_exc_balance, c = 1.1, JEE = g)
    return W

def generate_random_connectivity_matrix_unif(n_neurons: int, r_inhibitory: float, g: float, connectivity_proba: float,
                                        random_seed: int, balance_W: bool=True, inh_exc_balance: float=1):
    """ Generates random connectivity matrix

    Args:
        n_neurons: number of neurons
        r_inhibitory: ratio of inhibitory neurons
        g: gain
        connectivity_proba: probability of two neurons to be connected (directed)
        random_seed: random seed
        balance_W: whether to balance E and I weights on the input side of every neuron
        inh_exc_balance: how to balabce E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W: a balanced weight matrix
    """

    n_neurons_exc = int(n_neurons * (1 - r_inhibitory))

    # Random Erdos Renyi graph
    er_graph = nx.erdos_renyi_graph(n_neurons, p=connectivity_proba, seed=random_seed, directed=True)
    W_er = nx.to_numpy_array(er_graph)

    # Random log-normally distributed weights
    random_state = np.random.RandomState(random_seed)
    # W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons))) * W_er
    W = g * W_er

    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]

    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        # W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
        W = input_balance_quadrants(W, n_neurons_exc, gI=inh_exc_balance, c = 1.1, JEE = g)
    return W


def input_balance_quadrants(W: np.ndarray, n_neurons_exc: int, gI: float=1, c: float=1.1, JEE: float=1):
    """ Balances E and I in quadrants EE, II, EI, and IE

    Args:
        W: a weight matrix
        n_neurons_exc: number of excitatory neurons in the matrix (first n_neurons_exc are assumed to be the excitatory neurons)
        gI, c: how to balance E and I. The greater gI is, the stronger inhibition is relative to excitation,
        and the greater c, the more `cross talk' between I and E. For stability, need c > 1 and gI >= 1
        JEE: baseline parameter that establishes the strength of EE connections and scales up or down the strength of all connections

    Returns:
        W_bal: a balanced weight matrix
    """
    W_bal = W.copy()
    n_neurons_inh = len(W_bal) - n_neurons_exc

    #normalizing each quadrant and setting strength of EE to value given for JEE
    W_bal[:n_neurons_exc,:n_neurons_exc] =  JEE*W_bal[:n_neurons_exc,:n_neurons_exc]/(n_neurons_exc*np.mean(W_bal[:n_neurons_exc,:n_neurons_exc])) #EE
    print("EE mean: ", np.mean(W_bal[:n_neurons_exc,:n_neurons_exc]))
    W_bal[n_neurons_exc:,:n_neurons_exc] =  JEE*c*W_bal[n_neurons_exc:,:n_neurons_exc]/(n_neurons_exc*np.mean(W_bal[n_neurons_exc:,:n_neurons_exc])) #IE
    print("IE mean: ", np.mean(W_bal[n_neurons_exc:,:n_neurons_exc]))
    W_bal[:n_neurons_exc,n_neurons_exc:] =  -1*JEE*gI*c*W_bal[:n_neurons_exc,n_neurons_exc:]/(n_neurons_inh*np.mean(W_bal[:n_neurons_exc,n_neurons_exc:])) #EI
    print("EI mean: ", np.mean(W_bal[:n_neurons_exc,n_neurons_exc:]))
    W_bal[n_neurons_exc:,n_neurons_exc:] =  -1*JEE*gI*W_bal[n_neurons_exc:,n_neurons_exc:]/(n_neurons_inh*np.mean(W_bal[n_neurons_exc:,n_neurons_exc:])) #II
    print("II mean: ", np.mean(W_bal[n_neurons_exc:,n_neurons_exc:]))

    return W_bal

def complex_stability_radius(A):
    """
    Calculates the 2-norm complex stability radius of a matrix A.
    This is the same value as the smallest epsilon for which the eps-pseudospectrum intersects Re z = 0
    """
    n = A.shape[0]
    I = np.eye(n, dtype=complex)

    # Function to maximize: the 2-norm of the resolvent (which is the inverse of the smallest singular value)
    def resolvent_norm(omega):
        M = 1j * omega * I - A

        sigma_min = np.min(np.linalg.svd(M, compute_uv=False))
        return 1.0 / sigma_min

    res = minimize_scalar(lambda w: -resolvent_norm(w), bounds=(-1e3, 1e3), method='bounded')
    return 1.0 / (-res.fun)

def calc_eAint(A,c=0):
    """
    Calculates the point t_star at which pseudospectrum analysis guarantees || e^{At} ||_2 >= c,
    then approximates the integral of the matrix exponential from time 0 until t_star.

    Parameters:
    A - matrix of which exponential is being considered. For our use case, commonly W - I.
    c - constant >= 0, adjustable to scale final integration time, as described above.

    Returns:
    eAint - approximation via Simpson to int_0^{t_star}  || e^{At} ||_2 dt
    """

    tMax = 50
    ts = np.linspace(0,50,100)
    eAs = [np.linalg.norm(expm(A*ts[i]),ord=2) for i in range(len(ts))]
    eps = complex_stability_radius(A)
    print('eps: ' + str(eps))
    M = max(eAs)
    t_star = (1-c)/(eps*M)

    ts = np.linspace(0,t_star,round(t_star/.2))
    eAint = simpson([np.linalg.norm(expm(ts[i]*A),ord=2) for i in range(len(ts))])*(ts[1]-ts[0])

    return ts, eAint

def calc_eAvint(A,v,ts):
    #calculates int_0^{t_star}  || e^{At} v ||_2 dt for a given vector v and a matrix A
    eAvs = [np.linalg.norm(expm_multiply(A*ts[j], v),ord=2) for j in range(len(ts))]
    return simpson(eAvs)*(ts[1]-ts[0])


def rand_v_ints(A,N):
    #N = number of random vectors v
    ts, eAint = calc_eAint(A)
    print("done with first int")

    #N vectors of length matching A x 1
    eAv_ints = np.zeros((N,1))
    vs = []
    for i in range(N):
        v = np.random.randn(len(A))
        v = v/np.linalg.norm(v,ord=2)
        vs.append(v)
        eAv_ints[i] = calc_eAvint(A,v,ts)/eAint

    return eAv_ints, vs


def calc_vstar(A,c=0):
    """
    Calculates the point t_star at which pseudospectrum analysis guarantees || e^{At} ||_2 >= c,
    then finds the first right singular vector of the matrix exponential at that point.

    Parameters:
    A - matrix of which exponential is being considered. For our use case, commonly W - I.
    c - constant >= 0, adjustable to scale final integration time, as described above.

    Returns:
    v_star - first right singular vector of e^{At*}
    """

    tMax = 50
    ts = np.linspace(0,50,100)
    eAs = [np.linalg.norm(expm(A*ts[i]),ord=2) for i in range(len(ts))]
    eps = complex_stability_radius(A)
    print('eps: ' + str(eps))
    M = max(eAs)
    t_star = (1-c)/(eps*M)

    U, S, Vt = np.linalg.svd(expm(A*t_star))
    v_star = Vt[0, :]

    return v_star


def generate_stimulus(n_neurons: int, n_steps: int, r_inhibitory: float, stim_strength: float, t_input_start: float,
                      t_stimulus_end: float, dt: float):
    """ Generates stimulus array

    Args:
        n_neurons: Number of neurons
        n_steps: Number of steps
        r_inhibitory: Ratio of inhibitory neurons
        stim_strength: Stimulus strength
        t_input_start: Stimulus start time in s
        t_input_end: Stimulus end time in s
        dt: Time step

    Returns:
        stimulus, shape: [n_neurons, n_steps]
    """
    n_neurons_exc = int(n_neurons * (1 - r_inhibitory))

    stimulus = np.zeros([n_neurons, n_steps])

    t = np.arange(n_steps) * dt
    mask = np.logical_and(t >= t_stimulus_start, t <= t_stimulus_end)
    stimulus[:, mask] = stim_strength
    stimulus[n_neurons_exc:] = 0

    return stimulus

def transfer_func(x: np.ndarray, act_func: str="linear"):
    """ Applies an activation function the pre-activity.

    Args:
        pre_act: 'activity' before applying an activation function
        act_func: activiation function, 'relu' and 'linear' are supported
    """

    if act_func == "linear":
        z = x
    elif act_func == "relu":
        z = np.maximum(0, x)
    else:
        assert("Activation function not defined.")
    return z

def run_simulation(W: np.ndarray, stimulus: np.ndarray, dt: float=.01, tau: float=.25, act_func: str="relu"):
    """ Runs a simuluation using the Euler Method for a number of steps

    Args:
        W: a weight matrix
        stimulus: stimulus array
        dt: time step
        tau: time constant
        act_func: activiation function, 'relu' and 'linear' are supported

    Returns:
        r: firing rates
    """
    n_neurons = W.shape[0]
    n_steps = stimulus.shape[1]
    r = np.zeros([n_neurons, n_steps])

    for step in range(0, n_steps - 1):
        pre_act = np.matmul(W, r[:, step]) + stimulus[:, step]
        r[:, step+1] = r[:, step] + (-r[:, step] + transfer_func(pre_act, act_func=act_func)) * dt / tau

    return r


def filter_synapse_table(synapse_table: pd.DataFrame, pre_root_ids=None, post_root_ids=None):
    """Filter synapse table by pre and post root ids.

    Args:
        synapse_table: synapse table with pre_pt_root_ids and post_pt_root_ids as pd.DataFrame
        pre_root_ids: np.ndarray, list or pd.Series if root_ids to filter on the presynaptic side
        post_root_ids: np.ndarray, list or pd.Series if root_ids to filter on the postsynaptic side

    Returns:
        synapse_table: filtered synapse table
    """

    if pre_root_ids is not None:
        assert isinstance(pre_root_ids, (np.ndarray, list, pd.core.series.Series)), f"IDs have to be of type np.ndarray, list or pd.Series; got {type(pre_root_ids)}"
        pre_m = np.isin(synapse_table["pre_pt_root_id"], pre_root_ids)
    else:
        pre_m = np.ones(len(synapse_table), dtype=bool)

    if post_root_ids is not None:
        assert isinstance(post_root_ids, (np.ndarray, list, pd.core.series.Series)), f"IDs have to be of type np.ndarray, list or pd.Series; got {type(pre_root_ids)}"
        post_m = np.isin(synapse_table["post_pt_root_id"], post_root_ids)
    else:
        post_m = np.ones(len(synapse_table), dtype=bool)

    return synapse_table[pre_m & post_m]


def observed_plus_random_connectivity_matrix_random_inh(syn_mat: np.ndarray, r_inhibitory: float, g: float,
                                             connectivity_proba: float, random_seed: int, balance_W: bool=True,
                                             inh_exc_balance: float=1):
    """ Generates random connectivity matrix while using observed connectivity for EE portion

    Args:
        syn_mat: observed connectivity matrix
        r_inhibitory: ratio of inhibitory neurons
        g: gain
        connectivity_proba: probability of two neurons to be connected (directed)
        random_seed: random seed
        balance_W: whether to balance E and I weights on the input side of every neuron
        inh_exc_balance: how to balance E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W: a balanced weight matrix
    """
    n_neurons = int(len(syn_mat) / (1 - r_inhibitory))
    n_neurons_exc = len(syn_mat)

    # Random Erdos Renyi graph
    er_graph = nx.erdos_renyi_graph(n_neurons, p=connectivity_proba, seed=random_seed, directed=True)
    W_er = nx.to_numpy_array(er_graph)

    # EE mask
    EE_mask = np.zeros_like(W_er, dtype=bool)
    EE_mask[:n_neurons_exc, :n_neurons_exc] = True

    # Random log-normally distributed weights
    # random_state = np.random.RandomState(random_seed)
    random_state = np.random.default_rng(random_seed)
    W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons)))
    W[~EE_mask] *= W_er[~EE_mask]

    # Plug in observed matrix
    W[EE_mask] = syn_mat.T.flatten() * g

    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]

    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        # W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
        W = input_balance_quadrants(W,n_neurons_exc,gI=inh_exc_balance, c = 1.1, JEE = g)
    return W


#NOT FINISHED - FOR NOW USE RANDOM FN
def observed_plus_random_connectivity_matrix_semirandom_inh(syn_mat: np.ndarray, r_inhibitory: float, g: float,
                                             connectivity_proba: float, random_seed: int, balance_W: bool=True,
                                             inh_exc_balance: float=1):
    """ Generates random connectivity matrix while using observed connectivity for EE portion
    inhibitory connectivity is random in receiving connections but lightly selective in outgoing connections

    Args:
        syn_mat: observed connectivity matrix
        r_inhibitory: ratio of inhibitory neurons
        g: gain
        connectivity_proba: probability of two neurons to be connected (directed)
        random_seed: random seed
        balance_W: whether to balance E and I weights on the input side of every neuron
        inh_exc_balance: how to balance E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W: a balanced weight matrix
    """
    n_neurons = int(len(syn_mat) / (1 - r_inhibitory))
    n_neurons_exc = len(syn_mat)

    # Random Erdos Renyi graph
    er_graph = nx.erdos_renyi_graph(n_neurons, p=connectivity_proba, seed=random_seed, directed=True)
    W_er = nx.to_numpy_array(er_graph)

    # EE mask
    EE_mask = np.zeros_like(W_er, dtype=bool)
    EE_mask[:n_neurons_exc, :n_neurons_exc] = True

    # Random log-normally distributed weights
    # random_state = np.random.RandomState(random_seed)
    random_state = np.random.default_rng(random_seed)
    W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons)))
    W[~EE_mask] *= W_er[~EE_mask]

    # Plug in observed matrix
    W[EE_mask] = syn_mat.T.flatten() * g

    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]

    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        # W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
        W = input_balance_quadrants(W,n_neurons_exc,gI=inh_exc_balance, c = 1.1, JEE = g)
    return W


#test N random vectors at a limited distance from vstar
def rand_v_ints_vstar(A,N,r=0.1):
    #N = number of random vectors v
    ts, eAint = calc_eAint(A)
    print("done with first int")

    vstar = calc_vstar(A)

    #N vectors of length matching A x 1
    eAv_ints = np.zeros((N,1))
    vs = []
    rats = []
    for i in range(N):
        perturbation = np.random.randn(len(A))
        perturbation = r*perturbation/np.linalg.norm(perturbation)
        v = vstar + perturbation
        v = v/np.linalg.norm(v,ord=2)
        vs.append(v)
        eAv_ints[i] = calc_eAvint(A,v,ts)/eAint
        rats.append(eAv_ints[i]/np.linalg.norm(v - vstar))

    return eAv_ints, vs, rats

def rand_v_ints_neg_vstar(A,N,r=0.1):
    #N = number of random vectors v
    ts, eAint = calc_eAint(A)
    print("done with first int")

    vstar = -1*calc_vstar(A)

    #N vectors of length matching A x 1
    eAv_ints = np.zeros((N,1))
    vs = []

    for i in range(N):
        perturbation = np.random.randn(len(A))
        perturbation = r*perturbation/np.linalg.norm(perturbation)
        v = vstar + perturbation
        v = v/np.linalg.norm(v,ord=2)
        vs.append(v)
        eAv_ints[i] = calc_eAvint(A,v,ts)/eAint

    return eAv_ints, vs

def henrici_departure(A):
    """Henrici’s departure from normality"""
    fro_norm_sq = np.linalg.norm(A, ord="fro") ** 2
    eig_norm_sq = np.sum(np.abs(linalg.eigvals(A)) ** 2)
    return np.sqrt(fro_norm_sq - eig_norm_sq.real) / np.linalg.norm(A, ord="fro")