import os
os.chdir('..')
os.getcwd()


from common.bandits import *
from common.online_module import *
from matplotlib import pyplot as plt
import pandas as pd
import logging
import datetime
import pickle



import dataset
#files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501","../data/R6/ydata-fp-td-clicks-v1_0.20090502","../data/R6/ydata-fp-td-clicks-v1_0.20090503","../data/R6/ydata-fp-td-clicks-v1_0.20090504","../data/R6/ydata-fp-td-clicks-v1_0.20090505","../data/R6/ydata-fp-td-clicks-v1_0.20090506","../data/R6/ydata-fp-td-clicks-v1_0.20090507","../data/R6/ydata-fp-td-clicks-v1_0.20090508","../data/R6/ydata-fp-td-clicks-v1_0.20090509","../data/R6/ydata-fp-td-clicks-v1_0.20090510")
files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501","../data/R6/ydata-fp-td-clicks-v1_0.20090502","../data/R6/ydata-fp-td-clicks-v1_0.20090503","../data/R6/ydata-fp-td-clicks-v1_0.20090504","../data/R6/ydata-fp-td-clicks-v1_0.20090505")
dataset.get_yahoo_events(files)


print('File read complete!')

save_log_dir = 'log'
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
time_mark = datetime.datetime.today().strftime('%Y%m%d')
log_file = os.path.join(save_log_dir,'offline_model_test_%s.txt' % (time_mark))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Define params which include hyper-param
## Data param
events = dataset.events
item_features = dataset.features
n_arms = dataset.n_arms
for t, event in enumerate(events):
    user = event[2]
    break
n_user_features = len(user)
n_item_features = item_features.shape[1]
## hyper-param
hyper_param_list = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]


# Declare algorithm
algorithms = [
    ThompsonSampling(n_arms, False),
    ThompsonSampling(n_arms, True)
]
for hyper_param in hyper_param_list:
    algorithms.append(Ucb1(hyper_param, n_arms, False))
    algorithms.append(Ucb1(hyper_param, n_arms, True))

for hyper_param in hyper_param_list:
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "user", False))
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "both", False))
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "user", True))
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "user", False))
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "both", False))
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "user", True))
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "user", False))
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "both", False))
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "user", True))
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "both", True))



trial = np.repeat(0, len(algorithms))
cumulative_reward_list = [[]] * (len(algorithms))
mean_reward_list = [[]] * (len(algorithms))
for t, event in enumerate(events):
    # Get context
    user = event[2]
    pool_idx = event[3]
    pool_item_features = item_features[pool_idx,:]
    
    # Return offering
    displayed = event[0]
    offered = pool_idx[displayed]
    # Get Reward
    reward = event[1]
    
    for alg_idx in range(len(algorithms)):
        # temporary implement.
        ## recent alg
        try:
            cumulative_reward = cumulative_reward_list[alg_idx][-1]
            recent_mean_reward = mean_reward_list[alg_idx][-1]
        except IndexError:
            cumulative_reward = 0
            recent_mean_reward = 0
        chosen = algorithms[alg_idx].choose_arm(trial[alg_idx], user, pool_idx, pool_item_features)
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
                        
            algorithms[alg_idx].update(offered, reward, user, pool_idx, pool_item_features[chosen])
        else:
            if algorithms[alg_idx].update_option:
                chosen = displayed
                algorithms[alg_idx].update(offered, reward, user, pool_idx, pool_item_features[chosen])
                                
    if t % 100000 == 0:
        logger.info('###### %d th round complete!' % (t))

                

                
                
minimum_round = 0
for i in range(len(mean_reward_list)):
    if minimum_round ==0:
        minimum_round = len(mean_reward_list[i])
    else:
        minimum_round = min(minimum_round, len(mean_reward_list[i]))
        

alg_cate_list = []
alg_list = []
for i in range(len(mean_reward_list)):
    model = algorithms[i]
    alg_list.append(model.algorithm)
    if i == 0:
        bf_alg_cate = model.algorithm.split('_')[0]
        alg_cate_list.append(bf_alg_cate)
    else:
        if bf_alg_cate != model.algorithm.split('_')[0]:
            bf_alg_cate = model.algorithm.split('_')[0]
            alg_cate_list.append(bf_alg_cate)
    
    
#### plot Mean Reward
best_alg_cate_idx = []
for alg_cate_idx in range(len(alg_cate_list)):
    alg_cate = alg_cate_list[alg_cate_idx]
    best_perf = 0.
    alg_cate_idx = 0
    for i in range(len(mean_reward_list)):
        model = algorithms[i]
        if alg_cate == model.algorithm.split('_')[0]:
            plt.plot(mean_reward_list[i][:minimum_round], label="{}".format(model.algorithm))
            if best_perf < mean_reward_list[i][minimum_round-1]:
                best_perf = mean_reward_list[i][minimum_round-1]
                alg_cate_idx = i
    best_alg_cate_idx.append(alg_cate_idx)
    plt.title("Mean Reward for each model")
    plt.xlabel("T")
    plt.ylabel("Mean Reward")
    #plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    #plt.show()
    plt.savefig(os.path.join('Results','%s_MeanReward%s.png'%(alg_cate, time_mark)))

    
    
univ_offer_list = []
local_offer_list = []
for t, event in enumerate(events):
    # Get context
    user = event[2]
    pool_idx = event[3]
    pool_item_features = item_features[pool_idx,:]
    
    # Return offering
    displayed = event[0]
    offered = pool_idx[displayed]
    
    # Get data to make oracle model.
    univ_offer_list.append(offered)
    if t == 0:
        local_offer = [offered]
        bf_pool_idx = pool_idx
    else:
        if pool_idx == bf_pool_idx:
            local_offer.append(offered)
        else:
            local_offer_list.append(local_offer)
            local_offer = [offered]
            bf_pool_idx = pool_idx
            
            
univ_pd = pd.DataFrame({'offer' : univ_offer_list})
one_top_offer = univ_pd.value_counts().index[0][0]
univ_mean_reward = univ_pd.value_counts()[one_top_offer]/len(univ_offer_list)

local_mean_reward_list = []
for i in range(len(local_offer_list)):
    local_pd = pd.DataFrame({'offer' : local_offer_list[i]})
    one_top_offer = local_pd.value_counts().index[0][0]
    local_mean_reward = local_pd.value_counts()[one_top_offer]/len(local_offer_list[i])
    local_mean_reward_list.append(local_mean_reward)

local_mean_reward = np.array(local_mean_reward_list).mean()



for idx in best_alg_cate_idx:
    model = algorithms[idx]
    plt.plot(mean_reward_list[idx][:minimum_round], label="{}".format(model.algorithm.split('_')[0]))
    
plt.plot(np.repeat(univ_mean_reward,minimum_round), label = "univ_oracle")
plt.plot(np.repeat(local_mean_reward,minimum_round), label = "local_oracle")
plt.title("Mean Reward for each model")
plt.xlabel("T")
plt.ylabel("Mean Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('Results','Best_MeanReward%s.png'%(time_mark)))



import dill                           
filename = os.path.join('Results','globalsave.pkl')
dill.dump_session(filename)
