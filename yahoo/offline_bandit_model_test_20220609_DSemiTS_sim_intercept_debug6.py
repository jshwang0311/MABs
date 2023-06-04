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
import random

### Shuffle Data
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
log_file = os.path.join(save_log_dir,'offline_model_test_prop_int_debug6_%s.txt' % (time_mark))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



## hyper-param
#hyper_param_list=[0.01, 0.05, 0.1, 0.3]
hyper_param_list=[0.1, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.]
lints_hyper_param_list=[0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1]
#hyper_param= 0.1

lbd = 1.
gamma_list = [0.999]
gamma = 0.999


#n_iter = 10
n_iter = 2
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




    for hyper_param in hyper_param_list:
        algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, True))
    for hyper_param in hyper_param_list:
        algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, False))


    for hyper_param in lints_hyper_param_list:
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

    for hyper_param in hyper_param_list:
        js_model = SemiTS_Single_js(DIMENSION = DIMENSION, CONST_LAMBDA = 1.0, CONST_V = hyper_param)
        js_model.algorithm = 'SemiTSjsk'
        algorithms.append(js_model)



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
    

with open('Results/iter_cumulative_reward_list_debug6.pkl', 'wb') as f:
    pickle.dump(iter_cumulative_reward_list, f)
with open('Results/iter_mean_reward_list_debug6.pkl', 'wb') as f:
    pickle.dump(iter_mean_reward_list, f)
with open('Results/iter_last_mean_reward_debug6.pkl', 'wb') as f:
    pickle.dump(iter_last_mean_reward, f)

##################################################################
# 최초 load
##################################################################
proc_ind = 4
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
with open('Results/shuffle_event_v6.pkl', 'rb') as f:
    events = pickle.load(f)
with open('Results/item_features_v6.pkl', 'rb') as f:
    item_features = pickle.load(f)
with open('Results/n_arms_v6.pkl', 'rb') as f:
    n_arms = pickle.load(f)
with open('Results/n_user_features_v6.pkl', 'rb') as f:
    n_user_features = pickle.load(f)
with open('Results/n_item_features_v6.pkl', 'rb') as f:
    n_item_features = pickle.load(f)
    
    
save_log_dir = 'log'
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
time_mark = datetime.datetime.today().strftime('%Y%m%d')
log_file = os.path.join(save_log_dir,'offline_model_test_prop_int_debug6_afTunning_para%d.txt' % (proc_ind))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

with open('Results/iter_cumulative_reward_list_debug6.pkl' , 'rb') as f:
    iter_cumulative_reward_list = pickle.load(f)
with open('Results/iter_mean_reward_list_debug6.pkl' , 'rb') as f:
    iter_mean_reward_list = pickle.load(f)
with open('Results/iter_last_mean_reward_debug6.pkl' , 'rb') as f:
    iter_last_mean_reward = pickle.load(f)



hyper_param_list=[0.1, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.]
lints_hyper_param_list=[0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1]
#hyper_param= 0.1

lbd = 1.
gamma_list = [0.999]
gamma = 0.999


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




for hyper_param in hyper_param_list:
    algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, True))
for hyper_param in hyper_param_list:
    algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, False))


for hyper_param in lints_hyper_param_list:
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

for hyper_param in hyper_param_list:
    js_model = SemiTS_Single_js(DIMENSION = DIMENSION, CONST_LAMBDA = 1.0, CONST_V = hyper_param)
    js_model.algorithm = 'SemiTSjsk'
    algorithms.append(js_model)

##################################################################
##################################################################
##################################################################
# 돌았던 index 확인
##################################################################
log_file_ind = '1'
with open('Results/iter_cumulative_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    cumulative_reward_list = pickle.load(f)
with open('Results/iter_mean_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    mean_reward_list = pickle.load(f)
with open('Results/iter_last_mean_reward_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    last_mean_reward = pickle.load(f)

iter_cumulative_reward_list.append(cumulative_reward_list)
iter_mean_reward_list.append(mean_reward_list)
iter_last_mean_reward.append(last_mean_reward)


log_file_ind = '2'
with open('Results/iter_cumulative_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    cumulative_reward_list = pickle.load(f)
with open('Results/iter_mean_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    mean_reward_list = pickle.load(f)
with open('Results/iter_last_mean_reward_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    last_mean_reward = pickle.load(f)

iter_cumulative_reward_list.append(cumulative_reward_list)
iter_mean_reward_list.append(mean_reward_list)
iter_last_mean_reward.append(last_mean_reward)


log_file_ind = '3'
with open('Results/iter_cumulative_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    cumulative_reward_list = pickle.load(f)
with open('Results/iter_mean_reward_list_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    mean_reward_list = pickle.load(f)
with open('Results/iter_last_mean_reward_debug6_%s.pkl' % (log_file_ind), 'rb') as f:
    last_mean_reward = pickle.load(f)

iter_cumulative_reward_list.append(cumulative_reward_list)
iter_mean_reward_list.append(mean_reward_list)
iter_last_mean_reward.append(last_mean_reward)


total_last_mean_reward = np.repeat(0.,len(iter_last_mean_reward[0]))
for i in range(len(iter_last_mean_reward)):
    total_last_mean_reward += np.array(iter_last_mean_reward[i]).reshape(-1)
total_last_mean_reward = total_last_mean_reward/len(iter_last_mean_reward)    
########################################
# Extract best model
########################################
bf_model = ''
best_model_param_dict = {}
best_perf = 0.
for i in range(len(iter_cumulative_reward_list[0])):
    model = algorithms[i]
    #perf = mean_reward_list[i][len(mean_reward_list[i])-1]
    perf = total_last_mean_reward[i]
    try:
        param = model.v
    except:
        param = model.beta
    if bf_model != model.algorithm:
        if i > 0:
            best_model_param_dict[bf_model] = best_param
            print('%s : %.5f' % (bf_model, best_perf))
        best_param = param
        bf_model = model.algorithm
        best_perf = perf
    else:
        if perf > best_perf:
            best_param = param
            best_perf = perf
best_model_param_dict[bf_model] = best_param

'''
tunning_algorithms = algorithms
tunning_mean_reward_list = mean_reward_list
tunning_trial = trial
tunning_cumulative_reward_list = cumulative_reward_list
'''


######################################
## Simulate
######################################

algorithms = []
#algorithms = [ThompsonSampling(n_arms, False)]
algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, 
best_model_param_dict['DiscountedSemiparamLinTS_context_full_updateFalse_disjointFalse_adjustTrue'], 
gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = True))

        
algorithms.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, 
best_model_param_dict['DiscountedSemiparamLinTS_context_full_updateFalse_disjointFalse_adjustFalse'], 
gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = False))        


        
algorithms.append(Discounted_LinUCB_disjoint(n_user_features, n_item_features, gamma, lbd, 
best_model_param_dict['Discounted_LinUCB_context_full_updateFalse_disjointFalse'], 
n_arms, context="full", update_option = False, disjoint_option = False))        

'''
for i in range(len(algorithms)):
    model = algorithms[i]
    perf = mean_reward_list[i][len(mean_reward_list[i])-1]
    if model.algorithm == 'SemiparamLinTS_context_full_updateFalse_disjointFalse':
        print('%s %f : %.6f' % (str(model.adjust_nu), model.v ,perf))
'''

algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, 
best_model_param_dict['SemiparamLinTS_context_full_updateFalse_disjointFalse_adjustTrue'], n_arms, "full", False, False, True))
algorithms.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, 
best_model_param_dict['SemiparamLinTS_context_full_updateFalse_disjointFalse_adjustFalse'], n_arms, "full", False, False, False))

algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, 
best_model_param_dict['LinTS_context_full_updateFalse'],"full", False))
    
    
js_model = SemiTS_Single_js(DIMENSION = DIMENSION, CONST_LAMBDA = 1.0, CONST_V = best_model_param_dict['SemiTSjsk'])
js_model.algorithm = 'SemiTSjsk'
algorithms.append(js_model)
    


##trial = np.repeat(1, len(algorithms))
trial = np.repeat(0, len(algorithms))
cumulative_reward_list = [[]] * (len(algorithms))
mean_reward_list = [[]] * (len(algorithms))
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
        # temporary implement.
        ## recent alg
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
    if t % 10000000 == 0:
        with open('Results/Prop_sim_mean_reward_list_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'wb') as f:
            pickle.dump(mean_reward_list, f)

        # Save MAB model
        with open('model/Prop_sim_R6A_CMABs_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'wb') as f:
            pickle.dump(algorithms, f)

        



















with open('Results/Prop_sim_mean_reward_list_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'wb') as f:
    pickle.dump(mean_reward_list, f)

# Save MAB model
with open('model/Prop_sim_R6A_CMABs_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'wb') as f:
    pickle.dump(algorithms, f)



    
    
    
#######################################################################
# load and plot
#######################################################################
proc_ind = 2
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


with open('Results/Prop_sim_mean_reward_list_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'rb') as f:
    mean_reward_list = pickle.load(f)

with open('model/Prop_sim_R6A_CMABs_int_debug6_afTunning_para%d.pkl' % (proc_ind), 'rb') as f:
    algorithms = pickle.load(f)
                
minimum_round = 0
for i in range(len(mean_reward_list)):
    if minimum_round ==0:
        minimum_round = len(mean_reward_list[i])
    else:
        minimum_round = min(minimum_round, len(mean_reward_list[i]))
        


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

for idx in range(len(mean_reward_list)):
    try:
        model = algorithms[idx]
    except :
        model = js_model
    plt.plot(mean_reward_list[idx][:minimum_round], label="%s : %.5f" % (model.algorithm.split('_')[0], mean_reward_list[idx][minimum_round-1]))
    

plt.title("Mean Reward for each model")
plt.xlabel("T")
plt.ylabel("Mean Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
#plt.savefig(os.path.join('Results','Prop_sim_Best_MeanReward%s_int_debug5.png'%(time_mark)))
plt.show()





for idx in range(len(mean_reward_list)):
    try:
        model = algorithms[idx]
    except:
        model = js_model
    plt.plot(cumulative_reward_list[idx][:minimum_round], label="%s : %d" % (model.algorithm.split('_')[0], cumulative_reward_list[idx][minimum_round-1]))
    
plt.title("Cumulative Reward for each model")
plt.xlabel("T")
plt.ylabel("Cumulative Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
#plt.savefig(os.path.join('Results','Prop_sim_Best_CumulativeReward%s_int_debug5.png'%(time_mark)))
plt.show()

























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
    plt.savefig(os.path.join('Results','Prop_sim_%s_MeanReward%s_int_debug5.png'%(alg_cate, time_mark)))
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
plt.savefig(os.path.join('Results','Prop_sim_Best_MeanReward%s_int_debug5.png'%(time_mark)))
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
plt.savefig(os.path.join('Results','Prop_sim_Best_CumulativeReward%s_int_debug5.png'%(time_mark)))
plt.show()

