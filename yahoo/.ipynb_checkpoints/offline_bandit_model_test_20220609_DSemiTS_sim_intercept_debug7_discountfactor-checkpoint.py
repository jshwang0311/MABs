import os
os.chdir('..')
os.getcwd()
alg_idx = 2
####################################################################################################################################

from common.bandits import *
from common.online_module import *
from matplotlib import pyplot as plt
import pandas as pd
import logging
import datetime
import pickle

import dataset
with open('Results/shuffle_tunning_event_v6.pkl', 'rb') as f:
    events = pickle.load(f)
with open('Results/item_features_v6.pkl', 'rb') as f:
    item_features = pickle.load(f)
with open('Results/n_arms_v6.pkl', 'rb') as f:
    n_arms = pickle.load(f)
with open('Results/n_user_features_v6.pkl', 'rb') as f:
    n_user_features = pickle.load(f)
with open('Results/n_item_features_v6.pkl', 'rb') as f:
    n_item_features = pickle.load(f)

update_thres = 200000

print('File read complete!')

save_log_dir = 'log'
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
time_mark = datetime.datetime.today().strftime('%Y%m%d')
log_file = os.path.join(save_log_dir,'offline_model_test_prop_int_debug7_%s_%d.txt' % (time_mark, alg_idx))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




## hyper-param
hyper_param_list=[0.2]
#hyper_param= 0.1

lbd = 1.
gamma_list = [0.99, 0.9, 0.8, 0.7]
hyper_param_list=[0.1, 0.11, 0.13, 0.15, 0.2, 0.3, 0.5]



n_iter = 1
iter_trial = []
iter_cumulative_reward_list = []
iter_mean_reward_list = []
iter_last_mean_reward = []



for iter_idx in range(n_iter):
    # Declare algorithm
    algorithms = []

    for hyper_param in hyper_param_list:
        for gamma in gamma_list:
            algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = True))

    for hyper_param in hyper_param_list:
        for gamma in gamma_list:
            algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = False))
            
    for hyper_param in hyper_param_list:
        for gamma in gamma_list:
            algorithms.append(Discounted_LinUCB_disjoint(n_user_features, n_item_features, gamma, lbd, hyper_param, n_arms, context="full", update_option = False, disjoint_option = False))

    context = "full"
    hyper_param = 0.1
    if context == "user":
        DIMENSION = n_user_features
    elif context == "both":
        DIMENSION = n_user_features + n_item_features
    elif context == "user_interaction":
        DIMENSION = n_user_features + (n_user_features)* n_item_features
    elif context == "full":
        DIMENSION = n_user_features + n_item_features + (n_user_features)* n_item_features


    ##trial = np.repeat(1, len(algorithms))
    trial = np.repeat(0, len(algorithms))
    cumulative_reward_list = [[]] * (len(algorithms))
    mean_reward_list = [[]] * (len(algorithms))
    thres_alg_num = 0
    for t, event in enumerate(events):
        if len(np.where(trial>update_thres)[0]) >= len(algorithms):
            break
        # Get context
        user = event[2]
        pool_idx = event[3]
        pool_item_features = item_features[pool_idx,:]

        # Return offering
        pool_offered = event[0]
        offered = pool_idx[pool_offered]
        # Get Reward
        reward = event[1]

        user = user[:5]

        n_pool = len(pool_idx)
        ext_user = np.array([user] * n_pool)
        if context == 1:
            arms_features = ext_user
        elif context == 2:
            arms_features = np.hstack((ext_user, pool_item_features))
        elif (context == 3):
            interaction = np.einsum('ij,ik->ijk',ext_user[:,:(ext_user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            arms_features = np.hstack((ext_user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',ext_user[:,:(ext_user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            arms_features = np.hstack((ext_user, pool_item_features, interaction))
        
        for alg_idx in range(len(algorithms)):
            if trial[alg_idx] > update_thres:
                continue
            try:
                cumulative_reward = cumulative_reward_list[alg_idx][-1]
                recent_mean_reward = mean_reward_list[alg_idx][-1]
            except IndexError:
                cumulative_reward = 0
                recent_mean_reward = 0
            try:
                chosen = algorithms[alg_idx].choose_arm(trial[alg_idx], user, pool_idx, pool_item_features)
            except:
                chosen = algorithms[alg_idx].choose_arm(arms_features)

            if pool_idx[chosen] == offered:
                cumulative_reward += reward
                trial[alg_idx] += 1
                recent_cumulative_reward = cumulative_reward
                recent_trial = trial[alg_idx]
                recent_mean_reward = recent_cumulative_reward/recent_trial

                if len(cumulative_reward_list[alg_idx]) == 0:
                    cumulative_reward_list[alg_idx] = [cumulative_reward]
                    mean_reward_list[alg_idx] = [recent_mean_reward]
                else:
                    cumulative_reward_list[alg_idx].append(cumulative_reward)
                    mean_reward_list[alg_idx].append(recent_mean_reward)
                try:
                    algorithms[alg_idx].update(pool_offered, reward, user, pool_idx, pool_item_features)
                except:
                    algorithms[alg_idx].update(reward)
        if t % 100000 == 0:
            logger.info('###### %d th round complete!' % (t))
        if len(np.where(trial > update_thres)[0]) >= len(trial):
            print('Break iteration')
            break
    
    last_mean_reward = []
    for i in range(len(algorithms)):
        last_mean_reward.append(mean_reward_list[i][len(mean_reward_list[i])-1])
    iter_trial.append(trial)
    iter_cumulative_reward_list.append(cumulative_reward_list)
    iter_mean_reward_list.append(mean_reward_list)
    iter_last_mean_reward.append(last_mean_reward)
    

log_file_ind = log_file.split('.')[0].split('_')[-1]
with open('Results/iter_cumulative_reward_list_debug7_%s.pkl' % (log_file_ind), 'wb') as f:
    pickle.dump(iter_cumulative_reward_list, f)
with open('Results/iter_mean_reward_list_debug7_%s.pkl' % (log_file_ind), 'wb') as f:
    pickle.dump(iter_mean_reward_list, f)
with open('Results/iter_last_mean_reward_debug7_%s.pkl' % (log_file_ind), 'wb') as f:
    pickle.dump(iter_last_mean_reward, f)
