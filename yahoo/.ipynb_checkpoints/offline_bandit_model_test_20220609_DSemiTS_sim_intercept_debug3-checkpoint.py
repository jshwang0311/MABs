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

##0512 : DLinUCM param tunning
##0510 : modify input feature


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
log_file = os.path.join(save_log_dir,'offline_model_test_prop_int_debug3_%s.txt' % (time_mark))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Define params which include hyper-param
## Data param
events = dataset.events
item_features = dataset.features
# delete item intercept
item_features = item_features[:,:5]
n_arms = dataset.n_arms
for t, event in enumerate(events):
    user = event[2]
    break
n_user_features = len(user) - 1
n_item_features = item_features.shape[1]




## hyper-param
#hyper_param_list=[0.01, 0.05, 0.1, 0.3]
hyper_param_list=[0.1]
#hyper_param= 0.1

lbd = 1.
gamma_list = [0.999]


# Declare algorithm
algorithms = [ThompsonSampling(n_arms, False)]



for hyper_param in hyper_param_list:
    for gamma in gamma_list:
        algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = True))
        algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = False))


        
for hyper_param in hyper_param_list:
    for gamma in gamma_list:
        algorithms.append(Discounted_LinUCB_disjoint(n_user_features, n_item_features, gamma, lbd, hyper_param, n_arms, context="full", update_option = False, disjoint_option = False))

        


for hyper_param in hyper_param_list:
    algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, True))
    algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, False))
    
    
for hyper_param in hyper_param_list:
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "full", False))

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
js_model = SemiTS_Single_js(DIMENSION = DIMENSION, CONST_LAMBDA = 1.0, CONST_V = hyper_param)


##trial = np.repeat(1, len(algorithms))
trial = np.repeat(0, len(algorithms) + 1)
cumulative_reward_list = [[]] * (len(algorithms) + 1)
mean_reward_list = [[]] * (len(algorithms) + 1)
for t, event in enumerate(events):
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
        #trial[alg_idx] += 1
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
                        
            algorithms[alg_idx].update(pool_offered, reward, user, pool_idx, pool_item_features)
            
    n_pool = len(pool_idx)
    user = np.array([user] * n_pool)
    if context == 1:
        arms_features = user
    elif context == 2:
        arms_features = np.hstack((user, pool_item_features))
    elif (context == 3):
        interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
        arms_features = np.hstack((user, interaction))
    else:
        interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
        arms_features = np.hstack((user, pool_item_features, interaction))
    
    chosen = js_model.choose_arm(arms_features)
    alg_idx += 1
    #trial[alg_idx] += 1
    if pool_idx[chosen] == offered:
        try:
            cumulative_reward = cumulative_reward_list[alg_idx][-1]
            recent_mean_reward = mean_reward_list[alg_idx][-1]
        except IndexError:
            cumulative_reward = 0
            recent_mean_reward = 0
            
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
            
        js_model.update(reward)
        
                                
    if t % 100000 == 0:
        logger.info('###### %d th round complete!' % (t))




with open('Results/Prop_sim_mean_reward_list_int_debug3_%s.pkl' % (time_mark), 'wb') as f:
    pickle.dump(mean_reward_list, f)

# Save MAB model
with open('model/Prop_sim_R6A_CMABs_int_debug3_%s.pkl' % (time_mark), 'wb') as f:
    pickle.dump(algorithms, f)


                
                
minimum_round = 0
for i in range(len(mean_reward_list)):
    if minimum_round ==0:
        minimum_round = len(mean_reward_list[i])
    else:
        minimum_round = min(minimum_round, len(mean_reward_list[i]))
        

js_model.algorithm = 'SemiTSjsv'
alg_cate_list = []
alg_list = []
for i in range(len(mean_reward_list)):
    try:
        model = algorithms[i]
        alg_list.append(model.algorithm)
        if i == 0:
            bf_alg_cate = model.algorithm.split('_')[0]
            alg_cate_list.append(bf_alg_cate)
        else:
            if bf_alg_cate != model.algorithm.split('_')[0]:
                bf_alg_cate = model.algorithm.split('_')[0]
                alg_cate_list.append(bf_alg_cate)
    except:
        alg_list.append("SemiTSjsv")
        alg_cate_list.append("SemiTSjsv")
    
    

from pylab import rcParams

rcParams['figure.figsize'] = 25, 10
#### plot Mean Reward
best_alg_cate_idx = []
for alg_cate_idx in range(len(alg_cate_list)):
    alg_cate = alg_cate_list[alg_cate_idx]
    best_perf = 0.
    alg_cate_idx = 0
    for i in range(len(mean_reward_list)):
        try:
            model = algorithms[i]
        except:
            model = js_model
        if alg_cate == model.algorithm.split('_')[0]:
            plt.plot(mean_reward_list[i][:minimum_round], label="%s : %.3f" % (model.algorithm, mean_reward_list[i][minimum_round-1]))
            if best_perf < mean_reward_list[i][minimum_round-1]:
                best_perf = mean_reward_list[i][minimum_round-1]
                alg_cate_idx = i
    best_alg_cate_idx.append(alg_cate_idx)
    plt.title("Mean Reward for each model")
    plt.xlabel("T")
    plt.ylabel("Mean Reward")
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    #plt.show()
    plt.savefig(os.path.join('Results','Prop_sim_%s_MeanReward%s_int_debug3.png'%(alg_cate, time_mark)))
    plt.show()

    


for idx in best_alg_cate_idx:
    try:
        model = algorithms[idx]
    except :
        model = js_model
    plt.plot(mean_reward_list[idx][:minimum_round], label="%s : %.5f" % (model.algorithm.split('_')[0], mean_reward_list[idx][minimum_round-1]))
    
#plt.plot(np.repeat(univ_mean_reward,minimum_round), label = "univ_oracle : %.5f" % (univ_mean_reward))
#plt.plot(np.repeat(local_mean_reward,minimum_round), label = "local_oracle : %.5f" % (local_mean_reward))
plt.title("Mean Reward for each model")
plt.xlabel("T")
plt.ylabel("Mean Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('Results','Prop_sim_Best_MeanReward%s_int_debug3.png'%(time_mark)))
plt.show()





for idx in best_alg_cate_idx:
    try:
        model = algorithms[idx]
    except:
        model = js_model
    plt.plot(cumulative_reward_list[idx][:minimum_round], label="%s : %d" % (model.algorithm.split('_')[0], cumulative_reward_list[idx][minimum_round-1]))
    
#plt.plot(np.repeat(univ_mean_reward,minimum_round), label = "univ_oracle : %.5f" % (univ_mean_reward))
#plt.plot(np.repeat(local_mean_reward,minimum_round), label = "local_oracle : %.5f" % (local_mean_reward))
plt.title("Cumulative Reward for each model")
plt.xlabel("T")
plt.ylabel("Cumulative Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('Results','Prop_sim_Best_CumulativeReward%s_int_debug3.png'%(time_mark)))
plt.show()

