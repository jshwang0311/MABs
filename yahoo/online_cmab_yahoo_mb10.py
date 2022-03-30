import os
os.chdir('../')
import yahoo.dataset as dataset
from scipy.sparse import csr_matrix
import logging


files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090502","../data/R6/ydata-fp-td-clicks-v1_0.20090503","../data/R6/ydata-fp-td-clicks-v1_0.20090509")
#dataset.get_yahoo_events(files)
dataset.make_data_from_yahoo_events(files)


X = dataset.X
displays = dataset.displays
rewards = dataset.rewards
events = dataset.events
article_feat = dataset.features
nchoices = len(dataset.articles)



from sklearn.linear_model import SGDClassifier
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import LinUCB, AdaptiveGreedy, \
        SoftmaxExplorer, ActiveExplorer, EpsilonGreedy
from copy import deepcopy
import numpy as np


base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

## Metaheuristic using different base algorithms and configurations
linucb = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.3,
                ucb_from_empty = False, random_state = 1111)
### Important!!! the default hyperparameters for LinUCB in the reference paper
### are very different from what's used in this example
adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                        smoothing = None, beta_prior = ((3./nchoices,4.), 2),
                                        active_choice = 'weighted', decay_type = 'percentile',
                                        decay = 0.9997, batch_train = True,
                                        random_state = 2222)


linucb_both = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.3,
                ucb_from_empty = False, random_state = 1111)
### Important!!! the default hyperparameters for LinUCB in the reference paper
### are very different from what's used in this example
adaptive_active_greedy_both = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                        smoothing = None, beta_prior = ((3./nchoices,4.), 2),
                                        active_choice = 'weighted', decay_type = 'percentile',
                                        decay = 0.9997, batch_train = True,
                                        random_state = 2222)

models = [linucb, adaptive_active_greedy]
both_models = [linucb_both, adaptive_active_greedy_both]



rewards_lucb, rewards_aac = [list() for i in range(len(models))]

lst_rewards = [rewards_lucb, rewards_aac]

rewards_lucb_both, rewards_aac_both = [list() for i in range(len(both_models))]

lst_rewards_both = [rewards_lucb_both, rewards_aac_both]

# batch size - algorithms will be refit after N rounds
batch_size=10

    
# these lists will keep track of which actions does each policy choose
lst_a_lucb, lst_a_aac = [list() for i in range(len(models))]

lst_actions = [lst_a_lucb, lst_a_aac]

lst_a_lucb_both, lst_a_aac_both = [list() for i in range(len(both_models))]

lst_actions_both = [lst_a_lucb_both, lst_a_aac_both]




def extract_event_batch(events_batch, article_feat, batch_size):
    cand_article_num_list = []
    article_batch_add = []
    for i in range(len(events_batch)):
        cand_article_num_list.append(len(events_batch[i][1]))
        article_batch_add.append(article_feat[events_batch[i][1]])
    article_batch_add = np.concatenate(article_batch_add)
    return(cand_article_num_list, article_batch_add)

def extract_action_value(actions_cand_this_batch,action_st,action_end,events_batch_ind):
    action_value = []
    row_ind = np.arange(action_st,action_end)
    for i in range(len(row_ind)):
        action_value.append(actions_cand_this_batch[row_ind[i],events_batch_ind[i]])
    return(np.array(action_value))

def extract_actions(actions_cand_this_batch,events_batch):
    actions_this_batch = []
    if actions_cand_this_batch.shape[0]==len(events_batch):
        for i in range(len(events_batch)):
            actions_this_batch.append(np.array(events_batch[i][1])[np.argmax(actions_cand_this_batch[i,events_batch[i][1]])])
    else:
        action_st = 0
        for i in range(len(events_batch)):
            action_end = action_st + len(events_batch[i][1])
            action_value = extract_action_value(actions_cand_this_batch, action_st, action_end, events_batch[i][1])
            actions_this_batch.append(np.array(events_batch[i][1])[np.argmax(action_value)])
            action_st = action_end
    actions_this_batch = np.array(actions_this_batch).astype('uint8')
    return(actions_this_batch)


def extract_actions_add(actions_cand_this_batch,events_batch):
    actions_this_batch = []
    action_st = 0
    for i in range(len(events_batch)):
        action_end = action_st + len(events_batch[i][1])
        actions_this_batch.extend(list(np.array(events_batch[i][1])[np.argmax(actions_cand_this_batch[action_st:action_end,events_batch[i][1]],axis = 1)]))
        action_st = action_end
    actions_this_batch = np.array(actions_this_batch).astype('uint8')
    return(actions_this_batch)


# rounds are simulated from the full dataset
def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, rewards_batch, display_batch, events_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_cand_this_batch = model.decision_function(X_batch)
    actions_this_batch = extract_actions(actions_cand_this_batch,events_batch)
    #actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # rewards obtained now
    rewards_batch[actions_this_batch!=display_batch] = 0
    
    # keeping track of the sum of rewards received
    rewards.append(rewards_batch.sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)
    
    return new_actions_hist


# rounds are simulated from the full dataset
def simulate_rounds_stoch_bothcontext(model, rewards, actions_hist, X_batch_add, rewards_batch, rewards_batch_add, display_batch, events_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_cand_this_batch = model.decision_function(X_batch_add)
    actions_this_batch = extract_actions(actions_cand_this_batch,events_batch)
    actions_this_batch_add = extract_actions_add(actions_cand_this_batch,events_batch)
    #actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # rewards obtained now
    rewards_batch[actions_this_batch!=display_batch] = 0
    
    # keeping track of the sum of rewards received
    rewards.append(rewards_batch.sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch_add, actions_this_batch_add, rewards_batch_add)
    
    return new_actions_hist









save_log_dir = 'log'
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = os.path.join(save_log_dir,'yahoo_CMAB.txt')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size))):
    batch_st = (i) * batch_size
    batch_end = (i + 1) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])


    X_batch = X[batch_st:batch_end, :]
    display_batch = displays[batch_st:batch_end]
    rewards_batch = np.array(rewards[batch_st:batch_end])
    events_batch = events[batch_st:batch_end]

    cand_article_num_list, article_batch_add = extract_event_batch(events_batch, article_feat, batch_size)


    user_batch_add = np.repeat(X_batch, cand_article_num_list, axis=0)
    display_batch_add = np.repeat(display_batch, cand_article_num_list, axis=0)
    rewards_batch_add = np.repeat(rewards_batch, cand_article_num_list, axis=0)

    X_batch_add = np.hstack((user_batch_add, article_batch_add))
    
    for model_idx in range(len(models)):
        lst_actions[model_idx] = simulate_rounds_stoch(models[model_idx],
                                                   lst_rewards[model_idx],
                                                   lst_actions[model_idx],
                                                   X_batch, rewards_batch, display_batch, events_batch,
                                                   rnd_seed = batch_st)
    for model_idx in range(len(both_models)):
        lst_actions_both[model_idx] = simulate_rounds_stoch_bothcontext(both_models[model_idx],
                                                   lst_rewards_both[model_idx],
                                                   lst_actions_both[model_idx],
                                                   X_batch_add, rewards_batch, rewards_batch_add, display_batch, events_batch,
                                                   rnd_seed = batch_st)
    if (i % 100) == 0:
        logger.info('........%d/%d'% (i,int(np.floor(X.shape[0] / batch_size))))
    
    
    
min_len = min(len(rewards_lucb),len(rewards_aac), len(rewards_lucb_both),len(rewards_aac_both))
rewards_lucb = rewards_lucb[:min_len]
rewards_aac = rewards_aac[:min_len]
rewards_lucb_both = rewards_lucb_both[:min_len]
rewards_aac_both = rewards_aac_both[:min_len]

def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

mean_reward_lucb = get_mean_reward(rewards_lucb)
mean_rewards_aac = get_mean_reward(rewards_aac)
mean_reward_lucb_both = get_mean_reward(rewards_lucb_both)
mean_rewards_aac_both = get_mean_reward(rewards_aac_both)



csr_mat = csr_matrix((rewards[:min_len], displays[:min_len], list(range(min_len))), shape=(min_len-1, nchoices))







import matplotlib.pyplot as plt
from pylab import rcParams
%matplotlib inline



rcParams['figure.figsize'] = 25, 10
lwd = 5
cmap = plt.get_cmap('tab20')
colors=plt.cm.tab20(np.linspace(0, 1, 20))
rcParams['figure.figsize'] = 25, 10

ax = plt.subplot(111)
plt.plot(mean_reward_lucb, label="LinUCB (OLS)", linewidth=lwd,color=colors[0])
plt.plot(mean_rewards_aac, label="Adaptive Active Greedy (OLS)", linewidth=lwd,color=colors[16])
plt.plot(mean_reward_lucb_both, label="LinUCB with BothContext (OLS)", linewidth=lwd,color=colors[12])
plt.plot(mean_rewards_aac_both, label="Adaptive Active Greedy with BothContext (OLS)", linewidth=lwd,color=colors[15])
plt.plot(np.repeat(csr_mat.sum(0).max()/min_len,len(mean_reward_lucb)), label="Overall Best Arm (no context)",linewidth=lwd,color=colors[1],ls='dashed')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.5,
                 box.width, box.height * 3.])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':20})


plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks([i*5000 for i in range(46)],rotation=90)


plt.xlabel('Rounds (models were updated every 50 rounds)', size=30)
plt.ylabel('Cumulative Mean Reward', size=30)
plt.title('Comparison of Online Contextual Bandit Policies\n(Streaming-data mode)\n\nYahoo Data',size=30)
plt.grid()
save_fig_dir = 'Results'
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
plt.savefig(os.path.join(save_fig_dir,'CMR_mb'+str(batch_size)+'.png'), bbox_inches='tight')

######################################################################################################
ax = plt.subplot(111)
plt.plot(mean_reward_lucb, label="LinUCB (OLS)", linewidth=lwd,color=colors[0])
plt.plot(mean_rewards_aac, label="Adaptive Active Greedy (OLS)", linewidth=lwd,color=colors[16])
plt.plot(np.repeat(csr_mat.sum(0).max()/min_len,len(mean_reward_lucb)), label="Overall Best Arm (no context)",linewidth=lwd,color=colors[1],ls='dashed')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.5,
                 box.width, box.height * 3.])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':20})


plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks([i*5000 for i in range(46)],rotation=90)


plt.xlabel('Rounds (models were updated every 50 rounds)', size=30)
plt.ylabel('Cumulative Mean Reward', size=30)
plt.title('Comparison of Online Contextual Bandit Policies\n(Streaming-data mode)\n\nYahoo Data',size=30)
plt.grid()
save_fig_dir = 'Results'
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
plt.savefig(os.path.join(save_fig_dir,'CMR_sub_mb'+str(batch_size)+'.png'), bbox_inches='tight')