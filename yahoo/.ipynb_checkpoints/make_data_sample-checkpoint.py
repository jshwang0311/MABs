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
from sklearn.linear_model import LinearRegression
import copy

### Shuffle Data
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



model = LinearRegression()
model.fit(np_x, np_y)
model.predict(np_x)
beta_coef = model.coef_
beta0_coef = model.intercept_
print(beta_coef)
print(beta0_coef)







copy_model = copy.deepcopy(model)
sample_idx_num = int(len(copy_model.coef_)*0.6)
sample_idx = np.random.choice(np.arange(len(copy_model.coef_)), sample_idx_num, replace=False)
nonstationary = np.random.normal(0., 1., sample_idx_num)
copy_model.coef_[sample_idx] = copy_model.coef_[sample_idx] + nonstationary




print(copy_model.predict(example_x))
print(model.predict(example_x))


print(np.argmax(copy_model.predict(example_x)))
print(np.argmax(model.predict(example_x)))
