import sys
import gym
import pickle
import numpy as np
sys.path.append(sys.path[0] + "/..")
sys.path.append(".")
import gym_Toribash



from copy import deepcopy
from utils.tools import load_csv
from os.path import join
from hmmlearn import hmm
from stable_baselines import PPO1
from sklearn.decomposition import PCA




def create_policy_pca(Eg, n_comps=28):
    states = np.concatenate([eps for eps in Eg])
    pca = PCA(n_comps, whiten=True)
    return pca.fit_transform(states)


def create_policy_hmm(Eg, obs, components=12, iterations=15, tol=0.01, verbose=True):
    # create lengths of all of the matches
    # assumes generated games are all the same length
    lengths = Eg.shape[1]*np.ones(shape=Eg.shape[0]).astype(np.int)

    l = hmm.GaussianHMM(n_components=components, n_iter=iterations, verbose=verbose, tol=tol).fit(obs, lengths)
    return l


def create_expert_pca(p1,p2, n_comps=28):
    experts = np.concatenate((p1, p2), axis=1)
    pca = PCA(n_comps, whiten=True)
    pca.fit(experts)
    return pca.transform(experts)


def create_expert_hmm(df1, obs, components=12, iterations=15, tol=0.01, verbose=True):
    # get the lengths of all of the matches
    tmp = deepcopy(df1)
    tmp['match_length'] = 1
    func = {'match_length': lambda l: sum(l)}
    gb = tmp.groupby(by='match').agg(func)
    lengths = list(gb.match_length)

    # concat player 1 and player 2 
    l = hmm.GaussianHMM(n_components=components, n_iter=iterations, verbose=verbose, tol=tol).fit(obs, lengths)
    return l, lengths


def _stochastic_measure(model, env_dict, Pe_matches, hmm_expert, K = 47, T = 50, l = 10, pca_components=28, hmm_hidden_states=12, verbose=False):
    
    # load in model
    with open(env_dict, 'rb') as f:
        env_dict = pickle.load(f)
    env_id = env_dict['env_name']
    env = gym.make(env_id)
    env.init(**env_dict)
    h = PPO1.load(model)
    obs = env.reset()

    # generate some trajectories
    Eg = []
    for _ in range(K):
        tau = []
        for _ in range(T):
            action, _ = h.predict(obs)
            obs, _, dones, _ = env.step(action)
            tau.append(obs)
            if(dones):
                obs = env.reset()
        Eg.append(tau)

    env.close()
    # create second set of reduced trajectories
    # the shape is (K, T, 298)
    Eg = np.array(Eg).astype(np.float32)
    Pg = create_policy_pca(Eg, n_comps=pca_components)
    hmm_policy = create_policy_hmm(Eg, Pg, components=hmm_hidden_states, verbose=verbose)

    assert K == Pg.shape[0]//Eg.shape[1]

    Pg_matches = Pg.reshape((K, Eg.shape[1], Pg.shape[-1]))

    
    assert l < K
    pg_idx = np.random.choice(np.arange(len(Pg_matches)), size=(l,), replace=False)
    pe_idx = np.random.choice(np.arange(len(Pe_matches)), size=(l,), replace=False)

    score_val =  0
    for i in pe_idx:
        es = Pe_matches[i]
        for j in pg_idx:
            gs = Pg_matches[j]
            ee = np.exp(hmm_expert.score(es)/len(es))
            eg = np.exp(hmm_policy.score(es)/len(es))
            ge = np.exp(hmm_expert.score(gs)/len(gs))
            gg = np.exp(hmm_policy.score(gs)/len(gs))
            sig = np.sqrt((eg*ge)/(ee*gg))
            score_val += sig
    return score_val/l**2




def prepare_experts(pca_components=28, hidden_states=12, verbose=False):
    df1 = load_csv(join(sys.path[0], "../data/player1_state_info.csv"))
    df2 = load_csv(join(sys.path[0], "../data/player2_state_info.csv"))
    p1 = df1.drop(columns=['Unnamed: 0', 'match', 'turn'])
    p2 = df2.drop(columns=['Unnamed: 0', 'match', 'turn'])         
    
    obs = create_expert_pca(p1, p2, pca_components)
    hmm_expert, lengths = create_expert_hmm(df1, obs, components=hidden_states, verbose=verbose)

    counter = 0
    fixed_obs = []
    for i in lengths:
        fixed_obs.append(obs[counter:counter+i]) 
        counter = i - 1
    
    return hmm_expert, fixed_obs


def stochastic_measure(
    model, 
    env_dict, 
    num_generated_matches=47,
    num_samples=10,
    pca_dim = 28,
    hmm_hidden_states=12,
    Pe_matches=None, 
    hmm_expert=None,
    verbose=False):
    """
     Calculates the stochastic measure between expert matches and 
     a given model. 

     Args:
        model (string): path to the trained model \n

        env_dict (string): path to the environment for training the model (both model and env_dict \n
            are produced from model_trainer, mutli_limb_trainer, or hierarchy_trainer) \n

        num_generated_matches = 47 (int): number of matches to generate with the given model \n

        num_samples = 10 (int): number of samples to randomly choose from both the trained model \n
            and the expert matches. The final stochastic measure is averaged over num_samples^2. NOTE: \n
            num_samples < min(num_expert_matches, num_generated_matches) \n

        pca_dim = 28 (int): the dimension to reduce to using PCA for both the experts and the generated \n
            trajectories \n

        hmm_hidden_states = 12 (int): number of hidden states in the hidden markov model \n
        Pe_matches = None (list < np.array (Nx28) >): dimension-reduced expert matches where N \n
            is the length of each match and the length of the list is the number of expert matches \n
            used for calculated value. \n

        hmm_expert = None (hmmlearn.GaussianHMM): The hidden markov model trained on reduced observations \n
            from expert matches. If one is not given, it is generated and then returned along with the \n
            stochastic measure

    Return:
        metric (float): measure of similarity of the given model and the expert matches
        hmm (hmmlearn.GaussianHMM): hidden markov model trained using expert matches
        PCA_Expert_Matches (list <np.array (Nx28)>): dimension-reduced expert matches where N
            is the length of each match. 

    """
    # compare all of the models against a single learned model 
    # on the experts rather than retraining every single time
    if(Pe_matches is None or hmm_expert is None):
        hmm_expert, Pe_matches = prepare_experts(pca_components=pca_dim, hidden_states=hmm_hidden_states, verbose=verbose)
    
    return _stochastic_measure(model, env_dict, Pe_matches, hmm_expert, pca_components=pca_dim, hmm_hidden_states=hmm_hidden_states, verbose=verbose), hmm_expert, Pe_matches

if __name__ == "__main__":
    env_dict = join(sys.path[0], "../models/tight_range_continuous_actions_configs_dict.pkl")
    model = join(sys.path[0], "../models/tight_range_continuous_actions.pkl")
    print(stochastic_measure(model, env_dict)[0])