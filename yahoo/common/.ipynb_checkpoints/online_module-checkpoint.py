import numpy as np
import pickle
import os

class Online_module:
    """
    Online learning and evaluate part
    """

    def __init__(self, save_model_dir, model):
        """
        Parameters
        ----------
        algorithms : Contextual MAB algorithm list
        """
        self.model = model
        self.save_model_dir = save_model_dir
        
        alg_name = self.model.algorithm
        self.model_names = ['recent_' + alg_name, 'best_' + alg_name]
        for alg_fname in self.model_names:
            pickle.dump(self.model, open(os.path.join(self.save_model_dir, alg_fname), "wb"))
        
        self.trial = np.repeat(0, 2)
        self.cumulative_reward = [[]] * 2
        self.mean_reward = [[]] * 2
        

    def learn(self, user, pool_offered, reward, pool_idx, pool_item_features):
        chosen = self.model.choose_arm(self.trial[0], user, pool_idx, pool_item_features)
        if self.model.update_option:
            self.model.update(pool_offered, reward, user, pool_idx, pool_item_features)
            # save recent model
            pickle.dump(self.model, open(os.path.join(self.save_model_dir, self.model_names[0]), "wb"))
        else:
            if chosen == pool_offered:
                self.model.update(pool_offered, reward, user, pool_idx, pool_item_features)
                # save recent model
                pickle.dump(self.model, open(os.path.join(self.save_model_dir, self.model_names[0]), "wb"))
                
    
    
    def evaluate(self, user, pool_offered, reward, pool_idx, pool_item_features):
        # temporary implement.
        ## recent alg
        #recent_eval_alg = pickle.load(open(os.path.join(self.save_model_dir, self.model_names[0]), "rb"))
        recent_eval_alg = self.model
        try:
            cumulative_reward = self.cumulative_reward[0][-1]
            recent_mean_reward = self.mean_reward[0][-1]
        except IndexError:
            cumulative_reward = 0
            recent_mean_reward = 0
        chosen = recent_eval_alg.choose_arm(self.trial[0], user, pool_idx, pool_item_features)

        if chosen == pool_offered:
            cumulative_reward += reward
            self.trial[0] += 1
            recent_cumulative_reward = cumulative_reward
            recent_trial = self.trial[0]
            recent_mean_reward = cumulative_reward/recent_trial

            if len(self.cumulative_reward[0])==0:
                self.cumulative_reward[0] = [cumulative_reward]
                self.mean_reward[0] = [recent_mean_reward]
            else:
                self.cumulative_reward[0].append(cumulative_reward)
                self.mean_reward[0].append(recent_mean_reward)

        ## best alg
        best_eval_alg = pickle.load(open(os.path.join(self.save_model_dir, self.model_names[1]), "rb"))
        try:
            cumulative_reward = self.cumulative_reward[1][-1]
            best_mean_reward = self.mean_reward[1][-1]
        except IndexError:
            cumulative_reward = 0
            best_mean_reward = 0

        chosen = best_eval_alg.choose_arm(self.trial[1], user, pool_idx, pool_item_features)            
        if chosen == pool_offered:
            cumulative_reward += reward
            self.trial[1] += 1
            best_mean_reward = cumulative_reward/self.trial[1]

            if len(self.cumulative_reward[1])==0:
                self.cumulative_reward[1] = [cumulative_reward]
                self.mean_reward[1] = [best_mean_reward]
            else:
                self.cumulative_reward[1].append(cumulative_reward)
                self.mean_reward[1].append(best_mean_reward)

        ## Compare recent alg with best alg and if recent alg > best alg, then replace best alg with recent alg.
        if recent_mean_reward > best_mean_reward:
            pickle.dump(recent_eval_alg, open(os.path.join(self.save_model_dir, self.model_names[1]), "wb"))
            self.mean_reward[1] = [recent_mean_reward]
            self.cumulative_reward[1] = [recent_cumulative_reward]
            self.trial[1] = recent_trial
        
        

        # When understanding IPS estimator, implement.
        
        # Depoly best model
        

    