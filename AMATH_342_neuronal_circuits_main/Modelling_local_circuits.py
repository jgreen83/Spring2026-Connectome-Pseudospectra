##collecting functions from jupyter notebook so that I can call them from outside the folder

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns

# Numerical analysis
import numpy as np 

# Data handling
import pandas as pd
import collections

# Graph analysis and random graph generators
import networkx as nx

def plot_matrix(W: np.ndarray):
    """ Plots a connectivity matrix

    Args:
        W: a weight matrix

    """
    
    W_extreme_val = np.percentile(np.abs(W[W != 0]), 95) * 1.01
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    im = ax.imshow(W.T, vmin=-W_extreme_val, vmax=W_extreme_val, cmap="bwr")

    ax.set_xlabel("Postsynaptic neurons", fontsize=12)
    ax.set_ylabel("Presynaptic neurons", fontsize=12)

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
        W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
    return W


def calculate_time_to_before(r: np.ndarray, dt: float, stimulus: np.ndarray, threshold_ratio: float=.01):
    """ Calculates time for activity to reach a pre-stimulus level

    Args:
        r: rates, shape: [n_neurons, n_steps]
        dt: step size in s
        stimulus: stimulus, shape: [n_neurons, n_steps]
        threshold_ratio: determines the cutoff at which we assume the response to have returned to a pre-stimulus level
    Returns:
        Timepoint, delta T
    """
    
    stimulus_m = stimulus[0] > 0
    stimulus_start = np.where(stimulus_m)[0][0]
    stimulus_end = np.where(stimulus_m)[0][-1]
    
    stimulus_strength = np.mean(stimulus[0][stimulus_m])
    t = np.arange(r.shape[1]) * dt

    avg_act_after_stimulus_series = np.mean(r[:, stimulus_end:], axis=0)
    act_before_stimulus = np.mean(r[:, stimulus_start - 1])

    r_threshold =  stimulus_strength * threshold_ratio + act_before_stimulus
    T = t[stimulus_end:][avg_act_after_stimulus_series < r_threshold][0]
    return T, T - stimulus_end * dt
    
    
def plot_time_series(r: np.ndarray, stimulus: np.ndarray, dt: float, r_inhibitory: float):
    """ Plots the time series

    Args:
        r: rates, shape: [n_neurons, n_steps]
        stimulus: stimulus, shape: [n_neurons, n_steps]
        dt: step size in s
        r_inhibitory: ratio of inhibitory neurons

    """
    n_neurons = 200
    n_neurons_exc = int(n_neurons * (1 - r_inhibitory))
    t = np.arange(r.shape[1]) * dt

    try:
        t_to_before, t_to_before_delta = calculate_time_to_before(r[:n_neurons_exc], dt, stimulus[:n_neurons_exc])
    except:
        t_to_before, t_to_before_delta = None, None
    
    fig, ax = plt.subplots(3, figsize=(8, 8), dpi=150)
    
    ax[0].set_title("Excitatory neurons")
    ax[0].plot(t, r[:n_neurons_exc, :].T, lw=1, alpha=.25)
    ax[0].plot(t, np.mean(r[:n_neurons_exc], axis=0), lw=2, c="k")
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('r')

    if t_to_before is not None and t_to_before_delta > 0:
        ax[0].vlines(t_to_before, 0, np.max(r), ls="--", color="r")
        ax[0].text(t_to_before + 1, np.max(r)/2, f"$\Delta$t = {t_to_before_delta:.3f}s", color="r")

    ax[1].set_title("Inhibitory neurons")
    ax[1].plot(t, r[n_neurons_exc:, :].T, lw=1, alpha=.25)
    ax[1].plot(t, np.mean(r[n_neurons_exc:], axis=0), lw=2, c="k")
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('r')    
    
    ax[2].set_title("Stimulus signal to excitatory neurons")
    ax[2].plot(t, np.mean(stimulus[:n_neurons_exc], axis=0), lw=2, c="k")
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('Input')

    plt.tight_layout()
    plt.show()


def generate_stimulus(n_neurons: int, n_steps: int, r_inhibitory: float, stim_strength: float, t_stimulus_start: float, 
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


def generate_random_connectivity_matrix_FF(n_neurons: int, r_inhibitory: float, g: float, connectivity_proba: float, 
                                           random_seed: int, connectivity_proba_EE: float=None, make_EE_ff: bool=False, 
                                           balance_W: bool=True, inh_exc_balance: float=1):
    """ Generates random connectivity matrix

    Args:
        n_neurons: number of neurons
        r_inhibitory: ratio of inhibitory neurons
        g: gain
        connectivity_proba: probability of two neurons to be connected (directed)
        random_seed: random seed 
        connectivity_proba_EE: probability of two neurons to be connected (directed)
        balance_W: whether to balance E and I weights on the input side of every neuron
        inh_exc_balance: how to balance E and I. A value of 1 balances E and I equally; a lower value will weight E higher.

    Returns:
        W: a balanced weight matrix
    """    
    if connectivity_proba_EE is None:
        connectivity_proba_EE = connectivity_proba

    n_neurons_exc = int(n_neurons * (1 - r_inhibitory))

    # Random Erdos Renyi graph
    er_graph = nx.erdos_renyi_graph(n_neurons, p=connectivity_proba, seed=random_seed, directed=True)
    W_er = nx.to_numpy_array(er_graph)

    EE_er_graph = nx.erdos_renyi_graph(n_neurons_exc, p=connectivity_proba_EE, seed=random_seed, directed=True)
    W_EE_er = nx.to_numpy_array(EE_er_graph)

    ## EE mask
    EE_mask = np.zeros_like(W_er, dtype=bool)
    EE_mask[:n_neurons_exc, :n_neurons_exc] = True
    
    # Random log-normally distributed weights
    random_state = np.random.RandomState(random_seed)
    W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons)))
    W[~EE_mask] *= W_er[~EE_mask]
    W[EE_mask] *= W_EE_er.flatten()
    
    # Make EE feed-forward
    if make_EE_ff:
        not_EE_ff_mask = np.zeros_like(W, dtype=bool)
        not_EE_ff_mask[np.tril_indices_from(W, k=-1)] = True
        not_EE_ff_mask[~EE_mask] = False
        W[not_EE_ff_mask] = 0
    
    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]
    
    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
    return W


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



def observed_plus_random_connectivity_matrix(syn_mat: np.ndarray, r_inhibitory: float, g: float, 
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
    random_state = np.random.RandomState(random_seed)
    W = g * 10**np.abs(random_state.normal(1, .2, (n_neurons, n_neurons)))
    W[~EE_mask] *= W_er[~EE_mask]

    # Plug in observed matrix
    W[EE_mask] = syn_mat.T.flatten() * g

    # Enforce Dale's law
    W[:, n_neurons_exc:] = -1 * W[:, n_neurons_exc:]
    
    # Balance weights and ensure equal balance for all neurons
    if balance_W:
        W = input_balance_excitation_inhibition(W, n_neurons_exc, inh_exc_balance=inh_exc_balance)
    return W



def shuffle_W_weights(W: np.ndarray, random_seed: int):
    """ Shuffles the weights in a matrix
    
    This function does not change which neurons are connected, only the weight of the connection.

    Args:
        W: a weight matrix
        random_seed: random seed

    Returns:
        W_s: weight shuffled matrix

    """
    
    W_s = W.copy()
    mat_weights = W_s[W_s > 0].flatten()
    np.random.RandomState(random_seed).shuffle(mat_weights)
    W_s[W_s > 0] = mat_weights
    return W_s


def shuffle_W_conns(W, random_seed):
    """ Shuffles the connections in a matrix

    Args:
        W: a weight matrix
        random_seed: random seed

    Returns:
        W_s: shuffled weight matrix

    """
    W_s = W.copy()
    W_s = W_s.flatten()
    np.random.RandomState(random_seed).shuffle(W_s)
    W_s = W_s.reshape(W.shape)
    return W_s



