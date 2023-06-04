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
log_file = os.path.join(save_log_dir,'offline_model_test_%s.txt' % (time_mark))

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
n_user_features = len(user)
n_item_features = item_features.shape[1]




## hyper-param
#hyper_param_list = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
#hyper_param_list=[0.05,0.1]
hyper_param= 0.1


gamma = 1.
lbd = 1.
#gamma_list = [0.05, 0.1, 0.3]
#lbd_list = [0.05,0.1,0.3, 2.]


'''
####Debug
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
    break
    
gamma = 0.1
lbd = 0.1
delta = 0.1
#sigma = 1.
sigma = 0.01
L = 1.
S = 1.
alg2 = Discounted_LinUCB(n_user_features, n_item_features, gamma, lbd, delta, sigma, L, S, context="user", update_option = False)
self = alg2

beta = np.sqrt(self.lbd) * self.S + self.sigma * np.sqrt(2* np.log(1/self.delta) + self.n_features* np.log(1+(self.L**2 * (1-self.gamma ** (2*(t-1))))/(self.lbd* self.n_features * (1- self.gamma ** 2)) ))

n_pool = len(pool_idx)
user = np.array([user] * n_pool)
if self.context == 1:
    a_T = user
else:
    a_T = np.hstack((user, pool_item_features))


p = a_T @ self.theta + (beta* np.sqrt(np.diag(a_T @ np.linalg.inv(self.V) @ self.tilde_V @ np.linalg.inv(self.V) @ np.transpose(a_T)))).reshape(-1,1)

a = offered  # displayed article's index
mu_hat = (self.B_inv @ self.y).reshape(-1)
var = (self.v ** 2) * self.B_inv
mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)

n_pool = len(pool_idx)
user = np.array([user] * n_pool)
if self.context == 1:
    b_T = user
else:
    b_T = np.hstack((user, pool_item_features))

p = b_T @ mu_tilde.T
ac_mc = list(np.argmax(p,axis = 0))
pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
b_mean=np.dot(b_T.T,pi_est)

self.B += 2* (b_T[a,:] - b_mean).reshape(-1,1) @ (b_T[a,:] - b_mean).reshape(-1,1).T
self.B += 2* ((b_T.T @ np.diag(pi_est)) @ b_T) - 2*(b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
self.y += 2* reward* (b_T[a,:] - b_mean).reshape(-1,1)

B=B+2*np.outer(Bs_list[t][action]-b_mean,Bs_list[t][action]-b_mean)
B=B+2*np.dot(np.dot(np.transpose(Bs_list[t]),np.diag(pi_est)),Bs_list[t])-2*np.outer(b_mean,b_mean)
y=y+4*(Bs_list[t][action]-b_mean)*(-intercept+vs[t]+errors[t][action]+np.dot(Bs_list[t][action],mu))

##################
'''


# Declare algorithm
algorithms = [
    Discounted_LinUCB(n_user_features, n_item_features, gamma, lbd, hyper_param, context="both", update_option = False),
    LinUCB(n_user_features, n_item_features, hyper_param, "both", False)
]


trial = np.repeat(1, len(algorithms))
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
                        
            algorithms[alg_idx].update(pool_offered, reward, user, pool_idx, pool_item_features)
                                
    if t % 100000 == 0:
        logger.info('###### %d th round complete!' % (t))




with open('Results/mean_reward_list_%s.pkl' % (time_mark), 'wb') as f:
    pickle.dump(mean_reward_list, f)

# Save MAB model
with open('model/R6A_CMABs_%s.pkl' % (time_mark), 'wb') as f:
    pickle.dump(algorithms, f)


                
                
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
    
    

from pylab import rcParams

rcParams['figure.figsize'] = 25, 10

    


for idx in best_alg_cate_idx:
    model = algorithms[idx]
    plt.plot(mean_reward_list[idx][:minimum_round], label="%s : %.5f" % (model.algorithm.split('_')[0], mean_reward_list[idx][minimum_round-1]))
    
#plt.plot(np.repeat(univ_mean_reward,minimum_round), label = "univ_oracle : %.5f" % (univ_mean_reward))
#plt.plot(np.repeat(local_mean_reward,minimum_round), label = "local_oracle : %.5f" % (local_mean_reward))
plt.title("Mean Reward for each model")
plt.xlabel("T")
plt.ylabel("Mean Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('Results','Valid_DLinUCB_Best_MeanReward%s.png'%(time_mark)))
plt.show()





for idx in best_alg_cate_idx:
    model = algorithms[idx]
    plt.plot(cumulative_reward_list[idx][:minimum_round], label="%s : %d" % (model.algorithm.split('_')[0], cumulative_reward_list[idx][minimum_round-1]))
    
#plt.plot(np.repeat(univ_mean_reward,minimum_round), label = "univ_oracle : %.5f" % (univ_mean_reward))
#plt.plot(np.repeat(local_mean_reward,minimum_round), label = "local_oracle : %.5f" % (local_mean_reward))
plt.title("Cumulative Reward for each model")
plt.xlabel("T")
plt.ylabel("Cumulative Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join('Results','Valid_DLinUCB_Best_CumulativeReward%s.png'%(time_mark)))
plt.show()

#with open('Results/R6A_CMABs_%s.pkl' % (time_mark), 'rb') as f:
#    temp = pickle.load(f)
    
