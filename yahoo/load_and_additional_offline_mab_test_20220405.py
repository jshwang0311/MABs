import dill
import os

filename = os.path.join('../Results','globalsave.pkl')
dill.load_session(filename)
