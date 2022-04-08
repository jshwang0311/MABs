import numpy as np
import pickle
import os

class Online_module:
    """
    Online learning and evaluate part
    """

    def __init__(self, save_model_dir, algorithms):
        """
        Parameters
        ----------
        algorithms : Contextual MAB algorithm list
        """
        self.algorithms = algorithms
        self.save_model_dir = save_model_dir
        self.algorithms_name = []
        for i in range(len(self.algorithms)):
            alg_name = self.algorithms[i].algorithm
            alg_name_list = ['recent_' + alg_name, 'best_' + alg_name]
            for alg_fname in alg_name_list:
                pickle.dump(self.algorithms[i], open(os.path.join(self.save_model_dir, alg_fname), "wb"))

            self.algorithms_name.append(alg_name_list)
        self.trial = np.repeat(0, len(algorithms)*2)
        self.cumulative_reward = [[]] * (len(algorithms)*2)
        self.mean_reward = [[]] * (len(algorithms)*2)
        

    def learn(self, user, pool_offered, reward, pool_idx, pool_item_features):
        for alg_idx in range(len(self.algorithms)):
            chosen = self.algorithms[alg_idx].choose_arm(self.trial, user, pool_idx, pool_item_features)
            if self.algorithms[alg_idx].update_option:
                self.algorithms[alg_idx].update(pool_offered, reward, user, pool_idx, pool_item_features)
                # save recent model
                pickle.dump(self.algorithms[alg_idx], open(os.path.join(self.save_model_dir, self.algorithms_name[alg_idx][0]), "wb"))
            else:
                if chosen == pool_offered:
                    self.algorithms[alg_idx].update(pool_offered, reward, user, pool_idx, pool_item_features)
                    # save recent model
                    pickle.dump(self.algorithms[alg_idx], open(os.path.join(self.save_model_dir, self.algorithms_name[alg_idx][0]), "wb"))
                
    
    
    def evaluate(self, user, pool_offered, reward, pool_idx, pool_item_features):
        reward_idx = 0
        for alg_idx in range(len(self.algorithms)):
            # temporary implement.
            ## recent alg
            recent_eval_alg = pickle.load(open(os.path.join(self.save_model_dir, self.algorithms_name[alg_idx][0]), "rb"))
            try:
                cumulative_reward = self.cumulative_reward[reward_idx][-1]
                recent_mean_reward = self.mean_reward[reward_idx][-1]
            except IndexError:
                cumulative_reward = 0
                recent_mean_reward = 0
            chosen = recent_eval_alg.choose_arm(self.trial[reward_idx], user, pool_idx, pool_item_features)
            
            if chosen == pool_offered:
                cumulative_reward += reward
                self.trial[reward_idx] += 1
                recent_cumulative_reward = cumulative_reward
                recent_trial = self.trial[reward_idx]
                recent_mean_reward = cumulative_reward/recent_trial
                
                if len(self.cumulative_reward[reward_idx])==0:
                    self.cumulative_reward[reward_idx] = [cumulative_reward]
                    self.mean_reward[reward_idx] = [recent_mean_reward]
                else:
                    self.cumulative_reward[reward_idx].append(cumulative_reward)
                    self.mean_reward[reward_idx].append(recent_mean_reward)
            
            ## best alg
            reward_idx += 1
            best_eval_alg = pickle.load(open(os.path.join(self.save_model_dir, self.algorithms_name[alg_idx][1]), "rb"))
            try:
                cumulative_reward = self.cumulative_reward[reward_idx][-1]
                best_mean_reward = self.mean_reward[reward_idx][-1]
            except IndexError:
                cumulative_reward = 0
                best_mean_reward = 0

            chosen = best_eval_alg.choose_arm(self.trial[reward_idx], user, pool_idx, pool_item_features)            
            if chosen == pool_offered:
                cumulative_reward += reward
                self.trial[reward_idx] += 1
                best_mean_reward = cumulative_reward/self.trial[reward_idx]
                
                if len(self.cumulative_reward[reward_idx])==0:
                    self.cumulative_reward[reward_idx] = [cumulative_reward]
                    self.mean_reward[reward_idx] = [best_mean_reward]
                else:
                    self.cumulative_reward[reward_idx].append(cumulative_reward)
                    self.mean_reward[reward_idx].append(best_mean_reward)
            
            ## Compare recent alg with best alg and if recent alg > best alg, then replace best alg with recent alg.
            if recent_mean_reward > best_mean_reward:
                pickle.dump(recent_eval_alg, open(os.path.join(self.save_model_dir, self.algorithms_name[alg_idx][1]), "wb"))
                self.mean_reward[reward_idx] = [recent_mean_reward]
                self.cumulative_reward[reward_idx] = [recent_cumulative_reward]
                self.trial[reward_idx] = recent_trial
            
            reward_idx += 1
            # When understanding IPS estimator, implement.
        
        # Depoly best model
        

    