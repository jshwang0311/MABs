import os
os.chdir('..')
os.getcwd()


from common.bandits_intercept import *
from common.online_module import *
from common.functions import *
from matplotlib import pyplot as plt
import pandas as pd
import logging
import datetime
import pickle
import random
from sklearn.linear_model import LinearRegression
import copy


import dataset
files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501")
dataset.get_yahoo_events(files)
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

x_list = []
y_list = []
user_features_list = []
for t, event in enumerate(events):
    user = event[2]
    user = np.array(user[:5]).reshape(1,-1)
    user_features_list.append(user)
    pool_idx = event[3]

    # Return offering
    pool_offered = event[0]
    offered = pool_idx[pool_offered]
    pool_item_features = np.array(item_features[offered,:]).reshape(1,-1)
    # Get Reward
    reward = event[1]
    interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
    x = np.hstack((user, pool_item_features, interaction))
    x_list.append(x)
    y_list.append(reward)

user_features = np.concatenate(user_features_list, axis = 0)
np_x = np.concatenate(x_list, axis = 0)
np_y = np.array(y_list)

example_x = []
for i in range(item_features.shape[0]):
    pool_item_features = np.array(item_features[i,:]).reshape(1,-1)
    interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
    x = np.hstack((user, pool_item_features, interaction))
    example_x.append(x)
example_x = np.concatenate(example_x, axis = 0)



initial_model = LinearRegression()
initial_model.fit(np_x, np_y)
initial_model.predict(np_x)
beta_coef = initial_model.coef_
beta0_coef = initial_model.intercept_
print(beta_coef)
print(beta0_coef)

scale_beta_coef = beta_coef/np.sqrt(sum((beta_coef)**2) + beta0_coef**2)
scale_beta0_coef = beta0_coef/np.sqrt(sum((beta_coef)**2) + beta0_coef**2)

initial_model.coef_ = scale_beta_coef
initial_model.intercept_ = scale_beta0_coef


#perturb_ratio = 0.8
perturb_ratio = 0.9
total_model_num = 5
R_list = [0.1,1.0]
T = 10000
vs_option_list = [1,2,3,4,5,6]
simul_n = 3

model_list = []
model_list.append(initial_model)
for i in range(1,5):
    change_model = copy.deepcopy(initial_model)
    sample_idx_num = int(len(change_model.coef_)*perturb_ratio)
    sample_idx = np.random.choice(np.arange(len(change_model.coef_)), sample_idx_num, replace=False)
    change_model.coef_[sample_idx] = -1*change_model.coef_[sample_idx]
    model_list.append(change_model)


n_users = user_features.shape[0]
n_items = item_features.shape[0]
p_min = 0.
p_max = 1.
#########################################################################################################
######## hyperparam tunning
#########################################################################################################


for R in R_list:
    for vs_option in vs_option_list:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = f'log/log_vsoption{vs_option}_R{R}_perturb{perturb_ratio}.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('-----------------------------------------------------------------')
        logger.info('-----------------------------------------------------------------')

        Data_list = []
        for simul in range(simul_n):
            Data_list.append(makedata(n_users, n_items, R, T))

        #### model setting ####
        hyper_param_list=[0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
        gamma_list = [0.999, 0.99, 0.9, 0.7]
        lbd = 1.

        linTS_list = []
        for hyper_param in hyper_param_list:
            linTS_list.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "full", False))

        DSemiTS_list = []
        for hyper_param in hyper_param_list:
            for gamma in gamma_list:
                DSemiTS_list.append(Discounted_Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, gamma, lbd, n_arms, context="full", update_option = False, disjoint_option = False, adjust_nu = False))


        DLinUCB_list = []
        for hyper_param in hyper_param_list:
            for gamma in gamma_list:
                DLinUCB_list.append(Discounted_LinUCB_disjoint(n_user_features, n_item_features, gamma, lbd, hyper_param, n_arms, context="full", update_option = False, disjoint_option = False))


        SemiTS_list = []
        for hyper_param in hyper_param_list:
            SemiTS_list.append(Semiparam_LinTS_disjoint(n_user_features, n_item_features, hyper_param, n_arms, "full", False, False, False))


        cumulated_opt=list()
        cumulated_reward_linTS=list()
        cumulated_regret_linTS=list()
        for i in range(len(linTS_list)):
            cumulated_reward_linTS.append([])
            cumulated_regret_linTS.append([])

        cumulated_reward_DSemiTS=list()
        cumulated_regret_DSemiTS=list()
        for i in range(len(DSemiTS_list)):
            cumulated_reward_DSemiTS.append([])
            cumulated_regret_DSemiTS.append([])

        cumulated_reward_DLinUCB=list()
        cumulated_regret_DLinUCB=list()
        for i in range(len(DLinUCB_list)):
            cumulated_reward_DLinUCB.append([])
            cumulated_regret_DLinUCB.append([])


        cumulated_reward_SemiTS=list()
        cumulated_regret_SemiTS=list()
        for i in range(len(SemiTS_list)):
            cumulated_reward_SemiTS.append([])
            cumulated_regret_SemiTS.append([])







        for simul in range(simul_n):
            Data = Data_list[simul]
            CO=cumul_opt_v3(T, p_min, p_max, Data, model_list, user_features, item_features, vs_option)
            cumulated_opt.append(CO[0])
            vs=CO[1]
            for i in range(len(linTS_list)):
                mab_model = linTS_list[i]
                cumulated_reward_linTS[i].append(cumul_model_v2(T, Data, vs, model_list, mab_model, user_features, item_features))
                cumulated_regret_linTS[i].append(cumulated_opt[simul] - cumulated_reward_linTS[i][simul])

            for i in range(len(DSemiTS_list)):
                mab_model = DSemiTS_list[i]
                cumulated_reward_DSemiTS[i].append(cumul_model_v2(T, Data, vs, model_list, mab_model, user_features, item_features))
                cumulated_regret_DSemiTS[i].append(cumulated_opt[simul] - cumulated_reward_DSemiTS[i][simul])

            for i in range(len(DLinUCB_list)):
                mab_model = DLinUCB_list[i]
                cumulated_reward_DLinUCB[i].append(cumul_model_v2(T, Data, vs, model_list, mab_model, user_features, item_features))
                cumulated_regret_DLinUCB[i].append(cumulated_opt[simul] - cumulated_reward_DLinUCB[i][simul])

            for i in range(len(SemiTS_list)):
                mab_model = SemiTS_list[i]
                cumulated_reward_SemiTS[i].append(cumul_model_v2(T, Data, vs, model_list, mab_model, user_features, item_features))
                cumulated_regret_SemiTS[i].append(cumulated_opt[simul] - cumulated_reward_SemiTS[i][simul])


        with open(f'Results/linTS_cum_reward_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_reward_linTS, f)
        with open(f'Results/linTS_cum_regret_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_regret_linTS, f)

        with open(f'Results/DSemiTS_cum_reward_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_reward_DSemiTS, f)
        with open(f'Results/DSemiTS_cum_regret_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_regret_DSemiTS, f)

        with open(f'Results/DLinUCB_cum_reward_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_reward_DLinUCB, f)
        with open(f'Results/DLinUCB_cum_regret_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_regret_DLinUCB, f)

        with open(f'Results/SemiTS_cum_reward_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_reward_SemiTS, f)
        with open(f'Results/SemiTS_cum_regret_sim_sythetic_data_beta_l2_lt1_v{vs_option}_R{R}_perturb{perturb_ratio}.pkl', 'wb') as f:
            pickle.dump(cumulated_regret_SemiTS, f)


        ## optimal reward찾는 함수
        ## 각 모델별로 Data가 주어졌을때 simulation하는 함수
        Val_linTS = []
        for i in range(len(cumulated_regret_linTS)):
            Val_linTS.append(np.median(np.array(cumulated_regret_linTS[i])[:,-1]))

        Val_DSemiTS = []
        for i in range(len(cumulated_regret_DSemiTS)):
            Val_DSemiTS.append(np.median(np.array(cumulated_regret_DSemiTS[i])[:,-1]))

        Val_DLinUCB = []
        for i in range(len(cumulated_regret_DLinUCB)):
            Val_DLinUCB.append(np.median(np.array(cumulated_regret_DLinUCB[i])[:,-1]))

        Val_SemiTS = []
        for i in range(len(cumulated_regret_SemiTS)):
            Val_SemiTS.append(np.median(np.array(cumulated_regret_SemiTS[i])[:,-1]))

        logger.info('LinTS의 Regret 값 : %.4f' % min(Val_linTS))
        logger.info('SemiTS의 Regret 값 : %.4f' % min(Val_SemiTS))
        logger.info('DLinUCB의 Regret 값 : %.4f' % min(Val_DLinUCB))
        logger.info('DSemiTS의 Regret 값 : %.4f' % min(Val_DSemiTS))
        logger.removeHandler(file_handler)


