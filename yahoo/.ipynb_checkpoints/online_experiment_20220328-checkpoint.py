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
files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501","../data/R6/ydata-fp-td-clicks-v1_0.20090503")
#files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501", "../data/R6/ydata-fp-td-clicks-v1_0.20090502","../data/R6/ydata-fp-td-clicks-v1_0.20090503","../data/R6/ydata-fp-td-clicks-v1_0.20090504","../data/R6/ydata-fp-td-clicks-v1_0.20090509")
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
log_file = os.path.join(save_log_dir,'online_experiment_%s.txt' % (time_mark))

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
hyper_param = 0.1


# Declare algorithm
algorithms = [
    ThompsonSampling(n_arms, True)
]
for hyper_param in hyper_param_list:
    algorithms.append(Ucb1(hyper_param, n_arms, True))

for hyper_param in hyper_param_list:
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "user", True))
    algorithms.append(LinUCB(n_user_features, n_item_features, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "user", True))
    algorithms.append(LinearContextualThompsonSampling(n_user_features, n_item_features, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "user", True))
    algorithms.append(Disjoint_LinUCB(n_user_features, n_item_features, n_arms, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(Disjoint_LinTS(n_user_features, n_item_features, n_arms, hyper_param, "user", True))
    algorithms.append(Disjoint_LinTS(n_user_features, n_item_features, n_arms, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(Disjoint_LinTS(n_user_features, n_item_features, n_arms, hyper_param, "user", True))
    algorithms.append(Disjoint_LinTS(n_user_features, n_item_features, n_arms, hyper_param, "both", True))

for hyper_param in hyper_param_list:
    algorithms.append(Semiparam_LinTS(n_user_features, n_item_features, hyper_param, "user", True))
    algorithms.append(Semiparam_LinTS(n_user_features, n_item_features, hyper_param, "both", True))


save_model_dir = 'model'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

# Define online module
online_list = []
for i in range(len(algorithms)):
    online_list.append(Online_module(save_model_dir, algorithms[i]))

# Simulate
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
    
    for i in range(len(online_list)):
        # Call online module
        online_list[i].evaluate(user, pool_offered, reward, pool_idx, pool_item_features)
        online_list[i].learn(user, pool_offered, reward, pool_idx, pool_item_features)


        


# save and load binary model
pickle.dump(algorithms[3], open(os.path.join(save_model_dir, algorithms[3].algorithm), "wb"))
temp = pickle.load(open(os.path.join(save_model_dir, algorithms[3].algorithm), "rb"))

