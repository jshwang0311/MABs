import os
os.chdir('..')
os.getcwd()


from common.old_bandits import *
from common.evaluator import mod_evaluate
from matplotlib import pyplot as plt
import pandas as pd
import logging
import datetime




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
log_file = os.path.join(save_log_dir,'yahoo_testmodel_CMAB_%s.txt' % (time_mark))

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



## Simulate
algorithms = [LinUCB(0.1, context="both"), LinUCB(0.3, context="user"),Disjoint_LinUCB(0.3, context="both"), Disjoint_LinUCB(0.3, context="user"),
             LinearContextualThompsonSampling(0.3, context="both"),LinearContextualThompsonSampling(0.3, context="user"), Ucb1(0.1), ThompsonSampling()]

model_cumulative_reward = []
model_ind = 1
for model in algorithms:
    cumulative_reward = mod_evaluate(model, logger)
    model_cumulative_reward.append(cumulative_reward)
    model_ind = model_ind + 1


print('Training complete!')
minimum_round = 0
for i in range(len(model_cumulative_reward)):
    if minimum_round ==0:
        minimum_round = len(model_cumulative_reward[i])
    else:
        minimum_round = min(minimum_round, len(model_cumulative_reward[i]))
        
for i in range(len(model_cumulative_reward)):
    for j in range(len(model_cumulative_reward[i])):
        model_cumulative_reward[i][j] = model_cumulative_reward[i][j]/(j+1)

        
events = dataset.events
reward_list = []
for t, event in enumerate(events):
    reward = event[1]
    reward_list.append(reward)
        

#### plot Mean Reward
for i in range(len(model_cumulative_reward)):
    model = algorithms[i]
    plt.plot(model_cumulative_reward[i][:minimum_round], label="{}".format(model.algorithm))
        
plt.title("Mean Reward for each model")
plt.xlabel("T")
plt.ylabel("Mean Reward")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
#plt.savefig(os.path.join('Results','MeanRewardTotal%s.png'%(time_mark)))
plt.show()