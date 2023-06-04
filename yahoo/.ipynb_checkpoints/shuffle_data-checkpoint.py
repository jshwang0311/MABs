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
files = ("../data/R6/ydata-fp-td-clicks-v1_0.20090501","../data/R6/ydata-fp-td-clicks-v1_0.20090502","../data/R6/ydata-fp-td-clicks-v1_0.20090503","../data/R6/ydata-fp-td-clicks-v1_0.20090504","../data/R6/ydata-fp-td-clicks-v1_0.20090505","../data/R6/ydata-fp-td-clicks-v1_0.20090506","../data/R6/ydata-fp-td-clicks-v1_0.20090507","../data/R6/ydata-fp-td-clicks-v1_0.20090508","../data/R6/ydata-fp-td-clicks-v1_0.20090509","../data/R6/ydata-fp-td-clicks-v1_0.20090510")
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



random.shuffle(events)

tunning_events = events[:int(len(events)*(2/10))]

with open('Results/shuffle_event_v6.pkl', 'wb') as f:
    pickle.dump(events, f)
with open('Results/shuffle_tunning_event_v6.pkl', 'wb') as f:
    pickle.dump(tunning_events, f)
with open('Results/item_features_v6.pkl', 'wb') as f:
    pickle.dump(item_features, f)
with open('Results/n_arms_v6.pkl', 'wb') as f:
    pickle.dump(n_arms, f)
with open('Results/n_user_features_v6.pkl', 'wb') as f:
    pickle.dump(n_user_features, f)
with open('Results/n_item_features_v6.pkl', 'wb') as f:
    pickle.dump(n_item_features, f)