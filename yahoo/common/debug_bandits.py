import numpy as np
import time


class Discounted_Semiparam_LinTS_disjoint:
    """
    Contextual Multi-armed Bandit Algorithm for Semiparametric Reward Model (https://arxiv.org/pdf/1901.11221.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, v, gamma, lbd, n_arms, context="user", update_option = False, disjoint_option = False, adjust_nu = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        self.disjoint_option = disjoint_option
        self.n_arms = n_arms
        self.adjust_nu = adjust_nu
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            #self.n_features = n_user_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + (n_user_features)* n_item_features
        elif context == "full":
            self.context = 4
            #self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + n_item_features + (n_user_features)* n_item_features
        elif context == "user_hybrid":
            self.context = 5
            self.disjoint_n_features = n_user_features
            self.comb_n_features = (n_user_features)* n_item_features

        if self.context == 5:
            self.B_comb = lbd * np.array(np.identity(self.comb_n_features))
            self.B_inv_comb = (1/lbd) * np.array(np.identity(self.comb_n_features))
            self.B2_comb = lbd * np.array(np.identity(self.comb_n_features))
            self.y_comb = np.zeros((self.comb_n_features, 1))

            self.Bint_comb = lbd * np.array(np.identity(self.comb_n_features+1))
            self.Bint_inv_comb = (1/lbd) * np.array(np.identity(self.comb_n_features+1))
            self.yint_comb = np.zeros((self.comb_n_features+1, 1))
            
            
            self.B_disjoint = lbd * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.B_inv_disjoint = (1/lbd) * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.B2_disjoint = lbd * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.y_disjoint = np.zeros((self.n_arms, self.disjoint_n_features, 1))
            
            self.Bint_disjoint = lbd * np.array([np.identity(self.disjoint_n_features+1)] * self.n_arms)
            self.Bint_inv_disjoint = (1/lbd) * np.array([np.identity(self.disjoint_n_features+1)] * self.n_arms)
            self.yint_disjoint = np.zeros((self.n_arms, self.disjoint_n_features+1, 1))
            
            self.intercept_disjoint=np.zeros((self.n_arms,1))
            self.intercept_comb=0
            
        else:
            if self.disjoint_option:
                self.B = lbd * np.array([np.identity(self.n_features)] * self.n_arms)
                self.B_inv = (1/lbd) * np.array([np.identity(self.n_features)] * self.n_arms)
                self.B2 = lbd * np.array([np.identity(self.n_features)] * self.n_arms)
                self.y = np.zeros((self.n_arms, self.n_features, 1))
                
                self.Bint = lbd * np.array([np.identity(self.n_features+1)] * self.n_arms)
                self.Bint_inv = (1/lbd) * np.array([np.identity(self.n_features+1)] * self.n_arms)
                self.yint = np.zeros((self.n_arms, self.n_features+1, 1))
                self.intercept=np.zeros((self.n_arms,1))

            else:
                self.B = lbd * np.array(np.identity(self.n_features))
                self.B_inv = (1/lbd) * np.array(np.identity(self.n_features))
                self.B2 = lbd * np.array(np.identity(self.n_features))
                self.y = np.zeros((self.n_features, 1))
                
                self.Bint = lbd * np.array(np.identity(self.n_features + 1))
                self.Bint_inv = (1/lbd) * np.array(np.identity(self.n_features + 1))
                self.yint = np.zeros((self.n_features+1, 1))
                self.intercept=0
                
                
                
        self.v = v
        self.lbd = lbd
        self.gamma = gamma
        #self.algorithm = "DiscountedSemiparamLinTS_v" + str(self.v) + "_gm" + str(gamma) + "_lbd" + str(lbd) + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option)
        self.algorithm = "DiscountedSemiparamLinTS" + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option) + "_adjust" + str(adjust_nu)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
            
            
        if self.context == 5:
            B_inv = self.B_inv_disjoint[pool_idx]
            B2 = self.B2_disjoint[pool_idx]
            y = self.y_disjoint[pool_idx]
            mu_tilde_disjoint = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv[idx,:,:] @ B2[idx,:,:] @ B_inv[idx,:,:]
                mu_tilde_disjoint.append(np.random.multivariate_normal(mu_hat,var))
            #mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features, 1)
            mu_tilde_disjoint = np.vstack(mu_tilde_disjoint)
            
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            inv  = self.B_inv_comb @ self.B2_comb @ self.B_inv_comb
            var = (self.v ** 2) * inv
            mu_tilde_comb=np.random.multivariate_normal(mu_hat,var)
            mu_tilde_comb = np.repeat(mu_tilde_comb.reshape(1,-1), repeats = len(pool_idx),axis = 0)
            
            mu_tilde = np.expand_dims(np.concatenate((mu_tilde_disjoint, mu_tilde_comb), axis = 1), axis = 2)
            
            b_T = b_T.reshape(n_pool, self.disjoint_n_features + self.comb_n_features, 1)

            p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T
        else:
            if self.disjoint_option:
                B_inv = self.B_inv[pool_idx]
                B2 = self.B2[pool_idx]
                y = self.y[pool_idx]
                mu_tilde = []
                for idx in range(len(pool_idx)):
                    mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                    var = (self.v ** 2) * B_inv[idx,:,:] @ B2[idx,:,:] @ B_inv[idx,:,:]
                    mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
                mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)
                b_T = b_T.reshape(n_pool, self.n_features, 1)

                p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T
            else:
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                inv  = self.B_inv @ self.B2 @ self.B_inv
                var = (self.v ** 2) * inv
                mu_tilde=np.random.multivariate_normal(mu_hat,var)

                p = b_T @ mu_tilde
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        
        
        if self.context == 5:
            offered_item_features = pool_item_features[pool_offered]
            a = pool_idx[pool_offered]  # displayed article's index
            ###
            mu_hat = (self.B_inv_disjoint[a,:,:] @ self.y_disjoint[a,:,:]).reshape(-1)
            var = (self.v ** 2) * self.B_inv_disjoint[a,:,:]
            mu_tilde_disjoint=np.random.multivariate_normal(mu_hat,var,1000)
            
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            mu_tilde_comb=np.random.multivariate_normal(mu_hat,var,1000)

            mu_tilde = np.concatenate((mu_tilde_disjoint, mu_tilde_comb), axis = 1)
            ###            
            p = b_T @ mu_tilde.T
            ac_mc = list(np.argmax(p,axis = 0))
            pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
            b_mean=np.dot(b_T.T,pi_est)

            b_mean_disjoint = b_mean[:self.disjoint_n_features]
            b_mean_comb = b_mean[self.disjoint_n_features:]

            self.B_disjoint[a] = (1-self.gamma) * self.lbd * np.array(np.identity(self.disjoint_n_features)) +  self.gamma*self.B_disjoint[a]+ (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1) @ (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1).T
            self.B_disjoint[a] += ((b_T[:,:self.disjoint_n_features].T @ np.diag(pi_est)) @ b_T[:,:self.disjoint_n_features]) - (b_mean_disjoint.reshape(-1,1) @ b_mean_disjoint.reshape(-1,1).T)
            self.B_inv_disjoint[a] = np.linalg.inv(self.B_disjoint[a])

            self.B2_disjoint[a] = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.disjoint_n_features)) + (self.gamma**2) * self.B2_disjoint[a] + (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1) @ (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1).T
            self.B2_disjoint[a] += ((b_T[:,:self.disjoint_n_features].T @ np.diag(pi_est)) @ b_T[:,:self.disjoint_n_features]) - (b_mean_disjoint.reshape(-1,1) @ b_mean_disjoint.reshape(-1,1).T)
            
            
            self.B_comb = (1-self.gamma) * self.lbd * np.array(np.identity(self.comb_n_features)) + self.gamma* self.B_comb + (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1) @ (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1).T
            self.B_comb += ((b_T[:,self.disjoint_n_features:].T @ np.diag(pi_est)) @ b_T[:,self.disjoint_n_features:]) - (b_mean_comb.reshape(-1,1) @ b_mean_comb.reshape(-1,1).T)
            self.B_inv_comb = np.linalg.inv(self.B_comb)

            self.B2_comb = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.comb_n_features)) + (self.gamma**2) * self.B2_comb + (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1) @ (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1).T
            self.B2_comb += ((b_T[:,self.disjoint_n_features:].T @ np.diag(pi_est)) @ b_T[:,self.disjoint_n_features:]) - (b_mean_comb.reshape(-1,1) @ b_mean_comb.reshape(-1,1).T)
            

            if self.adjust_nu:
                self.y_disjoint[a] =  self.gamma*self.y_disjoint[a] + 2*(-self.intercept_disjoint[a]+reward) * (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1)

                newf=np.array([1.]+list(b_T[pool_offered,:self.disjoint_n_features]))
                self.Bint_disjoint[a] = self.Bint_disjoint[a] + np.outer(newf,newf)
                temp_B2inv=np.copy(self.Bint_inv_disjoint[a])
                self.Bint_inv_disjoint[a] =temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                self.yint_disjoint[a] =  self.yint_disjoint[a] + reward * newf.reshape(-1,1)
                self.intercept_disjoint[a] = np.matmul(self.Bint_inv_disjoint[a], self.yint_disjoint[a])[0]
                
                
                self.y_comb = self.gamma*self.y_comb +  2* (-self.intercept_comb + reward)* (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1)
                newf=np.array([1.]+list(b_T[pool_offered,self.disjoint_n_features:]))
                self.Bint_comb = self.Bint_comb + np.outer(newf,newf)
                temp_B2inv=np.copy(self.Bint_inv_comb)
                self.Bint_inv_comb=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                self.yint_comb =  self.yint_comb + reward * newf.reshape(-1,1)
                self.intercept_comb = np.matmul(self.Bint_inv_comb, self.yint_comb)[0]
            else:
                self.y_disjoint[a] = self.gamma*self.y_disjoint[a] + 2* reward* (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1)
                self.y_comb = self.gamma*self.y_comb +  2* reward* (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1)
            
        else:
            if self.disjoint_option:
                offered_item_features = pool_item_features[pool_offered]
                a = pool_idx[pool_offered]  # displayed article's index
                ###
                mu_hat = (self.B_inv[a,:,:] @ self.y[a,:,:]).reshape(-1)
                var = (self.v ** 2) * self.B_inv[a,:,:]
                mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                ###            
                p = b_T @ mu_tilde.T
                ac_mc = list(np.argmax(p,axis = 0))
                pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
                b_mean=np.dot(b_T.T,pi_est)


                self.B[a] = (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features)) +  self.gamma*self.B[a]+ (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B[a] += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                self.B_inv[a] = np.linalg.inv(self.B[a])

                self.B2[a] = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features)) + (self.gamma**2) * self.B2[a] + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B2[a] += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                
                if self.adjust_nu:
                    self.y[a] =  self.gamma*self.y[a] + 2*(-self.intercept[a]+reward) * (b_T[pool_offered,:] - b_mean).reshape(-1,1)

                    newf=np.array([1.]+list(b_T[pool_offered,:]))
                    self.Bint[a] = self.Bint[a] + np.outer(newf,newf)
                    temp_B2inv=np.copy(self.Bint_inv[a])
                    self.Bint_inv[a] =temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                    self.yint[a] =  self.yint[a] + reward * newf.reshape(-1,1)
                    self.intercept[a] = np.matmul(self.Bint_inv[a], self.yint[a])[0]
                else:
                    self.y[a] = self.gamma*self.y[a] + 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

            else:
                st_time = time.time()
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                inv  = self.B_inv @ self.B2 @ self.B_inv
                var = (self.v ** 2) * inv
                mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                print('sampling time :  %.5f' % (time.time() - st_time))
                st_time = time.time()
                p = b_T @ mu_tilde.T
                ac_mc = list(np.argmax(p,axis = 0))
                pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
                b_mean=np.dot(b_T.T,pi_est)


                self.B = (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features)) + self.gamma* self.B + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                
                self.B2 = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features)) + (self.gamma**2) * self.B2 + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B2 += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                print('mid time :  %.5f' % (time.time() - st_time))
                st_time = time.time()
                self.B_inv = np.linalg.inv(self.B)
                print('inv time :  %.5f' % (time.time() - st_time))
                
                
                if self.adjust_nu:
                    self.y =  self.gamma*self.y + 2*(-self.intercept+reward) * (b_T[pool_offered,:] - b_mean).reshape(-1,1)

                    newf=np.array([1.]+list(b_T[pool_offered,:]))
                    self.Bint = self.Bint + np.outer(newf,newf)
                    temp_B2inv=np.copy(self.Bint_inv)
                    self.Bint_inv=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                    self.yint =  self.yint + reward * newf.reshape(-1,1)
                    self.intercept = np.matmul(self.Bint_inv, self.yint)[0]
                else:
                    self.y = self.gamma*self.y +  2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)




class Discounted_Semiparam_LinTS:
    """
    Contextual Multi-armed Bandit Algorithm for Semiparametric Reward Model (https://arxiv.org/pdf/1901.11221.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, v, gamma, lbd, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            self.n_features = n_user_features + (n_user_features - 1)* n_item_features
        elif context == "full":
            self.context = 4
            self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features

        self.B = lbd * np.array(np.identity(self.n_features))
        self.B_inv = (1/lbd) * np.array(np.identity(self.n_features))
        self.B2 = lbd * np.array(np.identity(self.n_features))
        
        self.y = np.zeros((self.n_features, 1))
        self.v = v
        self.lbd = lbd
        self.gamma = gamma
        #self.algorithm = "DiscountedSemiparamLinTS_v" + str(self.v) + "_gm" + str(gamma) + "_lbd" + str(lbd) + "_context_" + context + '_update' + str(update_option)
        self.algorithm = "DiscountedSemiparamLinTS"+ "_context_" + context + '_update' + str(update_option)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        inv  = self.B_inv @ self.B2 @ self.B_inv
        var = (self.v ** 2) * inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var)
        
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        p = b_T @ mu_tilde
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        inv  = self.B_inv @ self.B2 @ self.B_inv
        var = (self.v ** 2) * inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))

        p = b_T @ mu_tilde.T
        ac_mc = list(np.argmax(p,axis = 0))
        pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
        b_mean=np.dot(b_T.T,pi_est)


        self.B = (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features)) + self.gamma* self.B + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
        self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        self.y = self.gamma*self.y +  2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

        self.B_inv = np.linalg.inv(self.B)
        
        self.B2 = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features)) + (self.gamma**2) * self.B2 + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
        self.B2 += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)



class Discounted_LinUCB:
    """
    Weighted Linear Bandits for Non-Stationary Environments (https://arxiv.org/pdf/1909.09146.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, gamma, lbd, beta, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            self.n_features = n_user_features + (n_user_features - 1)* n_item_features
        elif context == "full":
            self.context = 4
            self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features

        self.V = lbd * np.array(np.identity(self.n_features))
        self.tilde_V = lbd * np.array(np.identity(self.n_features))
        self.theta = np.zeros((self.n_features, 1))
        self.b = np.zeros((self.n_features, 1))
        
        self.gamma = gamma
        self.lbd = lbd
        self.beta = beta
        #self.algorithm = "Discounted_LinUCB_gm" + str(self.gamma) + "_lbd" + str(self.lbd) + "_context_" + context + '_update' + str(update_option)
        self.algorithm = "Discounted_LinUCB" + "_context_" + context + '_update' + str(update_option)
        '''
        self.delta = delta
        self.sigma = sigma
        self.S = S
        self.L = L
        self.algorithm = "Discounted_LinUCB_gm" + str(self.gamma) + "_lbd" + str(self.lbd) + "_dlt" + str(self.delta)  + '_sgma' + str(self.sigma) + '_S' + str(self.S) + '_L' + str(self.L) + "_context_" + context + '_update' + str(update_option)
        '''
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            a_T = user
        elif self.context == 2:
            a_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, pool_item_features, interaction))


        p = a_T @ self.theta + (self.beta* np.sqrt(np.diag(a_T @ np.linalg.inv(self.V) @ self.tilde_V @ np.linalg.inv(self.V) @ np.transpose(a_T)))).reshape(-1,1)
        
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            a_T = user
        elif self.context == 2:
            a_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, pool_item_features, interaction))
            
        self.V = self.gamma*self.V + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features))
        
        self.tilde_V = (self.gamma ** 2) * self.tilde_V + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features))
        
        self.b = self.gamma * self.b + reward * a_T[pool_offered,:].reshape(-1,1)
        self.theta = np.linalg.inv(self.V) @ self.b
        
        
class Discounted_LinUCB_disjoint:
    """
    Weighted Linear Bandits for Non-Stationary Environments (https://arxiv.org/pdf/1909.09146.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, gamma, lbd, beta, n_arms, context="user", update_option = False, disjoint_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        self.disjoint_option = disjoint_option
        self.n_arms = n_arms
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            #self.n_features = n_user_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + (n_user_features)* n_item_features
        elif context == "full":
            self.context = 4
            #self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + n_item_features + (n_user_features)* n_item_features
        elif context == "user_hybrid":
            self.context = 5
            self.disjoint_n_features = n_user_features
            self.comb_n_features = (n_user_features)* n_item_features

        if self.context == 5:
            self.V_disjoint = lbd * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.V_inv_disjoint = (1/lbd) * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.tilde_V_disjoint = lbd * np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.theta_disjoint = np.zeros((self.n_arms, self.disjoint_n_features, 1))
            self.b_disjoint = np.zeros((self.n_arms, self.disjoint_n_features, 1))        
        
            self.V_comb = lbd * np.array(np.identity(self.comb_n_features))
            self.V_inv_comb = (1/lbd) * np.array(np.identity(self.comb_n_features))
            self.tilde_V_comb = lbd * np.array(np.identity(self.comb_n_features))
            self.theta_comb = np.zeros((self.comb_n_features, 1))
            self.b_comb = np.zeros((self.comb_n_features, 1))
            
        else:
            if self.disjoint_option:
                self.V = lbd * np.array([np.identity(self.n_features)] * self.n_arms)
                self.V_inv = (1/lbd) * np.array([np.identity(self.n_features)] * self.n_arms)
                self.tilde_V = lbd * np.array([np.identity(self.n_features)] * self.n_arms)
                self.theta = np.zeros((self.n_arms, self.n_features, 1))
                self.b = np.zeros((self.n_arms, self.n_features, 1))        
            else:
                self.V = lbd * np.array(np.identity(self.n_features))
                self.V_inv = (1/lbd) * np.array(np.identity(self.n_features))
                self.tilde_V = lbd * np.array(np.identity(self.n_features))
                self.theta = np.zeros((self.n_features, 1))
                self.b = np.zeros((self.n_features, 1))
        
        self.gamma = gamma
        self.lbd = lbd
        self.beta = beta
        #self.algorithm = "Discounted_LinUCB_gm" + str(self.gamma) + "_lbd" + str(self.lbd) + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option)
        self.algorithm = "Discounted_LinUCB" + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            a_T = user
        elif self.context == 2:
            a_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, pool_item_features, interaction))
        
        if self.context == 5:
            a_T = a_T.reshape(n_pool, self.disjoint_n_features + self.comb_n_features, 1)
            a_T_disjoint = a_T[:,:self.disjoint_n_features,:]
            a_T_comb = a_T[:,self.disjoint_n_features:,:]
            theta = np.expand_dims(np.concatenate((self.theta_disjoint[pool_idx].reshape(len(pool_idx),self.disjoint_n_features), np.repeat(self.theta_comb.reshape(1,-1),repeats = len(pool_idx), axis =  0)), axis = 1), 2)
            
            V_inv_disjoint = self.V_inv_disjoint[pool_idx] @ self.tilde_V_disjoint[pool_idx] @ self.V_inv_disjoint[pool_idx]
            V_inv_comb = self.V_inv_comb @ self.tilde_V_comb @ self.V_inv_comb
            V_inv_comb = np.repeat(np.expand_dims(V_inv_comb, 0), repeats = len(pool_idx), axis = 0)
            
            mu = np.transpose(theta, (0, 2, 1)) @ a_T
            band = self.beta* np.sqrt(np.transpose(a_T_disjoint, (0,2,1)) @ V_inv_disjoint @ a_T_disjoint + np.transpose(a_T_comb, (0,2,1)) @ V_inv_comb @ a_T_comb)
            
            p = mu + band
            
        else:
            if self.disjoint_option:
                a_T = a_T.reshape(n_pool, self.n_features, 1)
                p = np.transpose(self.theta[pool_idx], (0, 2, 1)) @ a_T + (self.beta* np.sqrt(np.transpose(a_T, (0,2,1)) @ self.V_inv[pool_idx] @ self.tilde_V[pool_idx] @ self.V_inv[pool_idx] @ a_T))
            else:
                p = a_T @ self.theta + (self.beta* np.sqrt(np.diag(a_T @ self.V_inv @ self.tilde_V @ self.V_inv @ np.transpose(a_T)))).reshape(-1,1)
        
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            a_T = user
        elif self.context == 2:
            a_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((user, pool_item_features, interaction))
            
        
        if self.context == 5:
            offered_item_features = pool_item_features[pool_offered]
            a = pool_idx[pool_offered]  # displayed article's index
            
            self.V_disjoint[a] = self.gamma*self.V_disjoint[a] + a_T[pool_offered,:self.disjoint_n_features].reshape(-1,1) @ np.transpose(a_T[pool_offered,:self.disjoint_n_features].reshape(-1,1)) + (1-self.gamma) * self.lbd * np.array(np.identity(self.disjoint_n_features))

            self.V_inv_disjoint[a] = np.linalg.inv(self.V_disjoint[a])
            self.tilde_V_disjoint[a] = (self.gamma ** 2) * self.tilde_V_disjoint[a] + a_T[pool_offered,:self.disjoint_n_features].reshape(-1,1) @ np.transpose(a_T[pool_offered,:self.disjoint_n_features].reshape(-1,1)) + (1-self.gamma**2) * self.lbd * np.array(np.identity(self.disjoint_n_features))

            self.b_disjoint[a] = self.gamma * self.b_disjoint[a] + reward * a_T[pool_offered,:self.disjoint_n_features].reshape(-1,1)
            self.theta_disjoint[a] = self.V_inv_disjoint[a] @ self.b_disjoint[a]
            
            
            self.V_comb = self.gamma*self.V_comb + a_T[pool_offered,self.disjoint_n_features:].reshape(-1,1) @ np.transpose(a_T[pool_offered,self.disjoint_n_features:].reshape(-1,1)) + (1-self.gamma) * self.lbd * np.array(np.identity(self.comb_n_features))

            self.V_inv_comb = np.linalg.inv(self.V_comb)
            self.tilde_V_comb = (self.gamma ** 2) * self.tilde_V_comb + a_T[pool_offered,self.disjoint_n_features:].reshape(-1,1) @ np.transpose(a_T[pool_offered,self.disjoint_n_features:].reshape(-1,1)) + (1-self.gamma**2) * self.lbd * np.array(np.identity(self.comb_n_features))

            self.b_comb = self.gamma * self.b_comb + reward * a_T[pool_offered,self.disjoint_n_features:].reshape(-1,1)
            self.theta_comb = self.V_inv_comb @ self.b_comb
        else:
            if self.disjoint_option:
                offered_item_features = pool_item_features[pool_offered]
                a = pool_idx[pool_offered]  # displayed article's index

                self.V[a] = self.gamma*self.V[a] + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features))

                self.V_inv[a] = np.linalg.inv(self.V[a])
                self.tilde_V[a] = (self.gamma ** 2) * self.tilde_V[a] + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features))

                self.b[a] = self.gamma * self.b[a] + reward * a_T[pool_offered,:].reshape(-1,1)
                self.theta[a] = self.V_inv[a] @ self.b[a]

            else:
                self.V = self.gamma*self.V + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features))

                self.V_inv = np.linalg.inv(self.V)
                self.tilde_V = (self.gamma ** 2) * self.tilde_V + a_T[pool_offered,:].reshape(-1,1) @ np.transpose(a_T[pool_offered,:].reshape(-1,1)) + (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features))

                self.b = self.gamma * self.b + reward * a_T[pool_offered,:].reshape(-1,1)
                self.theta = self.V_inv @ self.b
        
        
        
        
class SemiTS_Single_js():
    def __init__(self, DIMENSION, CONST_LAMBDA, CONST_V):       
        self.dim = DIMENSION
        
        
        
        self.Lambda = CONST_LAMBDA
        self.v = CONST_V
        self.adjust_nu=True

        
        self.B=np.eye(self.dim)*self.Lambda
        self.B_inv=np.eye(self.dim)/self.Lambda
        self.y=np.zeros(self.dim)
        self.mu_hat=np.zeros(self.dim)
        

        self.B2 = np.eye(self.dim+1)*self.Lambda
        self.B2_inv = np.eye(self.dim+1)/self.Lambda
        self.y2 = np.zeros(self.dim+1)
        self.intercept=0
        
    
    def choose_arm(self, arms_features):
        N=len(arms_features)
        B_INV=self.B_inv
        self.V=(self.v ** 2) * B_INV
        try:
            mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)    
        except:
            mu_tilde = np.random.default_rng().multivariate_normal(mean = self.mu_hat, cov = self.V, method = 'cholesky')
        
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        N=len(self.arms_features)
        try:
            mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
        except:
            mu_mc=np.random.default_rng().multivariate_normal(self.mu_hat,self.V, 1000, method = 'cholesky')
            
        est_mc=list((np.dot(self.arms_features,mu_mc.T)).T) 
        ac_mc=list(np.argmax(est_mc,axis=1))
        pi_est=np.array([float(ac_mc.count(i))/len(ac_mc) for i in range(N)])
        b_mean=np.dot(np.transpose(np.array(self.arms_features)),pi_est)
        
        self.B = self.B + np.outer(self.chosen_arm_feature-b_mean, self.chosen_arm_feature-b_mean)
        self.B = self.B + np.dot(np.dot(np.transpose(self.arms_features),np.diag(pi_est)),self.arms_features)-np.outer(b_mean,b_mean)
        
        temp_Binv=np.copy(self.B_inv)
        temp_armf=np.copy(self.chosen_arm_feature-b_mean)
        self.B_inv=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
        for i in range(N):
            temp_Binv=np.copy(self.B_inv)
            temp_armf=np.sqrt(pi_est[i])*(self.arms_features[i]-b_mean)
            self.B_inv=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
        
        if self.adjust_nu:
            self.y =  self.y + 2*(-self.intercept+OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
        else:
            self.y =  self.y + 2*(OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
        
        self.mu_hat = np.matmul(self.B_inv, self.y)
        
        if self.adjust_nu:
            newf=np.array([1.]+list(self.chosen_arm_feature))
            self.B2 = self.B2 + np.outer(newf,newf)
            temp_B2inv=np.copy(self.B2_inv)
            self.B2_inv=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
            self.y2 =  self.y2 + OBSERVED_REWARD * newf
            self.intercept = np.matmul(self.B2_inv, self.y2)[0]


class Semiparam_LinTS:
    """
    Contextual Multi-armed Bandit Algorithm for Semiparametric Reward Model (https://arxiv.org/pdf/1901.11221.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, v, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            self.n_features = n_user_features + (n_user_features - 1)* n_item_features
        elif context == "full":
            self.context = 4
            self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.v = v
        #self.algorithm = "SemiparamLinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option)
        self.algorithm = "SemiparamLinTS" + "_context_" + context + '_update' + str(update_option)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        var = (self.v ** 2) * self.B_inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var)
        
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        p = b_T @ mu_tilde
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        var = (self.v ** 2) * self.B_inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))

        p = b_T @ mu_tilde.T
        ac_mc = list(np.argmax(p,axis = 0))
        pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
        b_mean=np.dot(b_T.T,pi_est)

        # Fix 20220412
        self.B += (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
        self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        self.y += 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)
        #self.B += 2*((b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T)
        #self.B += 2* ((b_T.T @ np.diag(pi_est)) @ b_T) - 2*(b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        #self.y += 4* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

        self.B_inv = np.linalg.inv(self.B)


        
class Semiparam_LinTS_disjoint:
    """
    Contextual Multi-armed Bandit Algorithm for Semiparametric Reward Model (https://arxiv.org/pdf/1901.11221.pdf)
    """
    
    def __init__(self, n_user_features, n_item_features, v, n_arms, context="user", update_option = False, disjoint_option = False, adjust_nu = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        self.disjoint_option = disjoint_option
        self.n_arms = n_arms
        self.adjust_nu = adjust_nu
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            #self.n_features = n_user_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + (n_user_features)* n_item_features
        elif context == "full":
            self.context = 4
            #self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + n_item_features + (n_user_features)* n_item_features
        elif context == "user_hybrid":
            self.context = 5
            self.disjoint_n_features = n_user_features
            self.comb_n_features = (n_user_features)* n_item_features

        if self.context == 5:
            self.B_comb = np.array(np.identity(self.comb_n_features))
            self.B_inv_comb = np.array(np.identity(self.comb_n_features))
            self.y_comb = np.zeros((self.comb_n_features, 1))

            self.B2_comb = np.array(np.identity(self.comb_n_features+1))
            self.B2_inv_comb = np.array(np.identity(self.comb_n_features+1))
            self.y2_comb = np.zeros((self.comb_n_features+1, 1))
            
            
            self.B_disjoint = np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.B_inv_disjoint = np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.y_disjoint = np.zeros((self.n_arms, self.disjoint_n_features, 1))
            
            self.B2_disjoint = np.array([np.identity(self.disjoint_n_features+1)] * self.n_arms)
            self.B2_inv_disjoint = np.array([np.identity(self.disjoint_n_features+1)] * self.n_arms)
            self.y2_disjoint = np.zeros((self.n_arms, self.disjoint_n_features+1, 1))
            
            self.intercept_disjoint=np.zeros((self.n_arms,1))
            self.intercept_comb=0
        else:            
            if disjoint_option:
                self.B = np.array([np.identity(self.n_features)] * self.n_arms)
                self.B_inv = np.array([np.identity(self.n_features)] * self.n_arms)
                self.y = np.zeros((self.n_arms, self.n_features, 1))

                self.B2 = np.array([np.identity(self.n_features+1)] * self.n_arms)
                self.B2_inv = np.array([np.identity(self.n_features+1)] * self.n_arms)
                self.y2 = np.zeros((self.n_arms, self.n_features+1, 1))
                self.intercept=np.zeros((self.n_arms,1))
            else:
                self.B = np.array(np.identity(self.n_features))
                self.B_inv = np.array(np.identity(self.n_features))
                self.y = np.zeros((self.n_features, 1))

                self.B2 = np.array(np.identity(self.n_features+1))
                self.B2_inv = np.array(np.identity(self.n_features+1))
                self.y2 = np.zeros((self.n_features+1, 1))
                self.intercept=0
            
        self.v = v
        #self.algorithm = "SemiparamLinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option)
        self.algorithm = "SemiparamLinTS" + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option) + "_adjust" + str(adjust_nu)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5) :
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        if self.context == 5:
            B_inv_disjoint = self.B_inv_disjoint[pool_idx]
            y_disjoint = self.y_disjoint[pool_idx]
            mu_tilde_disjoint = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv_disjoint[idx,:,:] @ y_disjoint[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv_disjoint[idx,:,:]
                mu_tilde_disjoint.append(np.random.multivariate_normal(mu_hat,var))
            #mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features, 1)
            mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features)

            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            mu_tilde_comb=np.random.multivariate_normal(mu_hat,var)
            #mu_tilde_comb = np.repeat(mu_tilde_comb.reshape(1,-1,1), repeats = len(pool_idx),axis = 0)
            mu_tilde_comb = np.repeat(mu_tilde_comb.reshape(1,-1), repeats = len(pool_idx),axis = 0)
            mu_tilde = np.expand_dims(np.concatenate((mu_tilde_disjoint, mu_tilde_comb), axis = 1), axis = 2)
            

            b_T = b_T.reshape(n_pool, self.disjoint_n_features + self.comb_n_features, 1)
            p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T
        else:
            if self.disjoint_option:
                B_inv = self.B_inv[pool_idx]
                y = self.y[pool_idx]
                mu_tilde = []
                for idx in range(len(pool_idx)):
                    mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                    var = (self.v ** 2) * B_inv[idx,:,:]
                    mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
                mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)


                b_T = b_T.reshape(n_pool, self.n_features, 1)
                p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T

            else:
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                var = (self.v ** 2) * self.B_inv
                mu_tilde=np.random.multivariate_normal(mu_hat,var)
                p = b_T @ mu_tilde

        
        
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
            
        
        if self.context == 5:
            offered_item_features = pool_item_features[pool_offered]
            a = pool_idx[pool_offered]  # displayed article's index
            ###
            mu_hat = (self.B_inv_disjoint[a,:,:] @ self.y_disjoint[a,:,:]).reshape(-1)
            var = (self.v ** 2) * self.B_inv_disjoint[a,:,:]
            mu_tilde_disjoint=np.random.multivariate_normal(mu_hat,var,1000)
            
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            mu_tilde_comb=np.random.multivariate_normal(mu_hat,var,1000)

            mu_tilde = np.concatenate((mu_tilde_disjoint, mu_tilde_comb), axis = 1)
            ###            
            p = b_T @ mu_tilde.T
            ac_mc = list(np.argmax(p,axis = 0))
            pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
            b_mean=np.dot(b_T.T,pi_est)

            b_mean_disjoint = b_mean[:self.disjoint_n_features]
            b_mean_comb = b_mean[self.disjoint_n_features:]
            
            # Fix 20220412
            self.B_disjoint[a] += (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1) @ (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1).T
            self.B_disjoint[a] += ((b_T[:,:self.disjoint_n_features].T @ np.diag(pi_est)) @ b_T[:,:self.disjoint_n_features]) - (b_mean_disjoint.reshape(-1,1) @ b_mean_disjoint.reshape(-1,1).T)
            self.B_inv_disjoint[a] = np.linalg.inv(self.B_disjoint[a])            
            
            self.B_comb += (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1) @ (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1).T
            self.B_comb += ((b_T[:,self.disjoint_n_features:].T @ np.diag(pi_est)) @ b_T[:,self.disjoint_n_features:]) - (b_mean_comb.reshape(-1,1) @ b_mean_comb.reshape(-1,1).T)
            self.B_inv_comb = np.linalg.inv(self.B_comb)
            
            
            
            if self.adjust_nu:
                self.y_disjoint[a] +=  2*(-self.intercept_disjoint[a]+reward) * (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1)
                
                newf=np.array([1.]+list(b_T[pool_offered,:self.disjoint_n_features]))
                self.B2_disjoint[a] = self.B2_disjoint[a] + np.outer(newf,newf)
                temp_B2inv=np.copy(self.B2_inv_disjoint[a])
                self.B2_inv_disjoint[a] =temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                self.y2_disjoint[a] =  self.y2_disjoint[a] + reward * newf.reshape(-1,1)
                self.intercept_disjoint[a] = np.matmul(self.B2_inv_disjoint[a], self.y2_disjoint[a])[0]
                
                
                self.y_comb +=  2*(-self.intercept_comb+reward) * (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1)
                
                newf=np.array([1.]+list(b_T[pool_offered,self.disjoint_n_features:]))
                self.B2_comb = self.B2_comb + np.outer(newf,newf)
                temp_B2inv=np.copy(self.B2_inv_comb)
                self.B2_inv_comb=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                self.y2_comb =  self.y2_comb + reward * newf.reshape(-1,1)
                self.intercept_comb = np.matmul(self.B2_inv_comb, self.y2_comb)[0]
            else:
                self.y_comb += 2* reward* (b_T[pool_offered,self.disjoint_n_features:] - b_mean_comb).reshape(-1,1)
                self.y_disjoint[a] += 2* reward* (b_T[pool_offered,:self.disjoint_n_features] - b_mean_disjoint).reshape(-1,1)
        else:
            if self.disjoint_option:
                offered_item_features = pool_item_features[pool_offered]
                a = pool_idx[pool_offered]  # displayed article's index
                ###
                mu_hat = (self.B_inv[a,:,:] @ self.y[a,:,:]).reshape(-1)
                var = (self.v ** 2) * self.B_inv[a,:,:]
                mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                ###            
                p = b_T @ mu_tilde.T
                ac_mc = list(np.argmax(p,axis = 0))
                pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
                b_mean=np.dot(b_T.T,pi_est)

                # Fix 20220412
                self.B[a] += (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B[a] += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)

                self.B_inv[a] = np.linalg.inv(self.B[a])            
                if self.adjust_nu:
                    self.y[a] +=  2*(-self.intercept[a]+reward) * (b_T[pool_offered,:] - b_mean).reshape(-1,1)

                    newf=np.array([1.]+list(b_T[pool_offered,:]))
                    self.B2[a] = self.B2[a] + np.outer(newf,newf)
                    temp_B2inv=np.copy(self.B2_inv[a])
                    self.B2_inv[a] =temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                    self.y2[a] =  self.y2[a] + reward * newf.reshape(-1,1)
                    self.intercept[a] = np.matmul(self.B2_inv[a], self.y2[a])[0]
                else:
                    self.y[a] += 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)


            else:
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                var = (self.v ** 2) * self.B_inv
                mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)

                p = b_T @ mu_tilde.T
                ac_mc = list(np.argmax(p,axis = 0))
                pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
                b_mean=np.dot(b_T.T,pi_est)

                # Fix 20220412
                self.B += (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                self.B_inv = np.linalg.inv(self.B)



                if self.adjust_nu:
                    self.y +=  2*(-self.intercept+reward) * (b_T[pool_offered,:] - b_mean).reshape(-1,1)

                    newf=np.array([1.]+list(b_T[pool_offered,:]))
                    self.B2 = self.B2 + np.outer(newf,newf)
                    temp_B2inv=np.copy(self.B2_inv)
                    self.B2_inv=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
                    self.y2 =  self.y2 + reward * newf.reshape(-1,1)
                    self.intercept = np.matmul(self.B2_inv, self.y2)[0]
                else:
                    self.y += 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

                
        
        
        



class Disjoint_LinUCB:
    """
    Disjoint_LinUCB algorithm implementation (https://arxiv.org/pdf/1003.0146.pdf)
    """

    def __init__(self, n_user_features, n_item_features, n_arms, alpha, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        n_arms : number
            total number of arms
        alpha : number
            Disjoint LinUCB hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        self.n_arms = n_arms
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            self.n_features = n_user_features + (n_user_features - 1)* n_item_features
        elif context == "full":
            self.context = 4
            self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features

        self.A = np.array([np.identity(self.n_features)] * self.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * self.n_arms)
        self.b = np.zeros((self.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        #self.algorithm = "DisjointLinUCB_alpha"+str(self.alpha)+"_context_" + context + '_update' + str(update_option)
        self.algorithm = "DisjointLinUCB"+"_context_" + context + '_update' + str(update_option)

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        elif self.context == 2:
            x = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            x = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            x = np.hstack((user, pool_item_features, interaction))

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        offered_item_features = pool_item_features[pool_offered]
        a = pool_idx[pool_offered]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        elif self.context == 2:
            x = np.hstack((user, offered_item_features))
        elif self.context == 3:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            x = np.hstack((user, interaction))
            x = x.reshape(-1)
        else:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            x = np.hstack((user, offered_item_features, interaction))
            x = x.reshape(-1)
            
        x = x.reshape((self.n_features, 1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, n_user_features, n_item_features, alpha, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        alpha : number
            LinUCB hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            self.n_features = n_user_features + (n_user_features - 1)* n_item_features
        elif context == "full":
            self.context = 4
            self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.alpha = alpha
        #self.algorithm = "LinUCB_alpha" + str(self.alpha) + "_context_" + context + '_update' + str(update_option)
        self.algorithm = "LinUCB" + "_context_" + context + '_update' + str(update_option)

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        mu_hat = self.B_inv @ self.y
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        #user = np.array(user)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        U = b_T @ mu_hat + self.alpha * np.diag(np.sqrt(b_T @ self.B_inv @ np.transpose(b_T))).reshape(-1,1)
        return np.argmax(U)
        


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        offered_item_features = pool_item_features[pool_offered]
        a = pool_idx[pool_offered]  # displayed article's index
        if self.context == 1:
            b = np.array(user)
        elif self.context == 2:
            b = np.hstack((user, offered_item_features))
        elif self.context == 3:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, interaction))
            b = b.reshape(-1)
        else:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, offered_item_features, interaction))
            b = b.reshape(-1)

        b = b.reshape((self.n_features, 1))

        self.B += b @ b.T
        self.y += reward * b
        self.B_inv = np.linalg.inv(self.B)
        
    
    
class Disjoint_LinTS:
    """
    Disjoint Linear contextual TS algorithm implementation (Bayesian version of Disjoint LinUCB)
    """

    def __init__(self, n_user_features, n_item_features, n_arms, v, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        n_arms : number
            total number of arms
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        self.n_arms = n_arms
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            #self.n_features = n_user_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + (n_user_features)* n_item_features
        elif context == "full":
            self.context = 4
            #self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + n_item_features + (n_user_features)* n_item_features
        elif context == "user_hybrid":
            self.context = 5
            self.disjoint_n_features = n_user_features
            self.comb_n_features = (n_user_features)* n_item_features

        if self.context == 5:
            self.B_disjoint = np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.B_inv_disjoint = np.array([np.identity(self.disjoint_n_features)] * self.n_arms)
            self.y_disjoint = np.zeros((self.n_arms, self.disjoint_n_features, 1))
            
            self.B_comb = np.array(np.identity(self.comb_n_features))
            self.B_inv_comb = np.array(np.identity(self.comb_n_features))
            self.y_comb = np.zeros((self.comb_n_features, 1))
            
            self.v = v
        else:
            self.B = np.array([np.identity(self.n_features)] * self.n_arms)
            self.B_inv = np.array([np.identity(self.n_features)] * self.n_arms)
            self.y = np.zeros((self.n_arms, self.n_features, 1))
            self.v = v
        #self.algorithm = "DisjointLinTS_v"+str(self.v)+"_context_" + context + '_update' + str(update_option)
        self.algorithm = "DisjointLinTS"+"_context_" + context + '_update' + str(update_option)
        
    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        
        
        if self.context == 5:
            b_T = b_T.reshape(n_pool, self.disjoint_n_features + self.comb_n_features, 1)
            B_inv = self.B_inv_disjoint[pool_idx]
            y = self.y_disjoint[pool_idx]
            mu_tilde_disjoint = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv[idx,:,:]
                try:
                    mu_tilde_disjoint.append(np.random.multivariate_normal(mu_hat,var))
                except:
                    mu_tilde_disjoint.append(np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky'))
                    
            #mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features, 1)
            mu_tilde_disjoint = np.vstack(mu_tilde_disjoint)
            
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            mu_tilde_comb=np.random.multivariate_normal(mu_hat,var)
            mu_tilde_comb = np.repeat(mu_tilde_comb.reshape(1,-1), repeats = len(pool_idx),axis = 0)
            mu_tilde = np.expand_dims(np.concatenate((mu_tilde_disjoint, mu_tilde_comb), axis = 1), axis = 2)
            p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T            
        else:
            b_T = b_T.reshape(n_pool, self.n_features, 1)
            n_pool = len(pool_idx)
            B_inv = self.B_inv[pool_idx]
            y = self.y[pool_idx]
            mu_tilde = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv[idx,:,:]
                try:
                    mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
                except:
                    mu_tilde.append(np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky'))
            mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)
            p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T
        
        return np.argmax(p)

    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        offered_item_features = pool_item_features[pool_offered]
        a = pool_idx[pool_offered]  # displayed article's index
        if self.context == 1:
            b = np.array(user)
        elif self.context == 2:
            b = np.hstack((user, offered_item_features))
        elif (self.context == 3 or self.context == 5):
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, interaction))
            b = b.reshape(-1)
        else:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, offered_item_features, interaction))
            b = b.reshape(-1)
        
        if self.context == 5:
            b = b.reshape((self.disjoint_n_features + self.comb_n_features, 1))

            self.B_disjoint[a] += b[:self.disjoint_n_features,:] @ b[:self.disjoint_n_features,:].T
            self.y_disjoint[a] += reward * b[:self.disjoint_n_features,:]
            self.B_inv_disjoint[a] = np.linalg.inv(self.B_disjoint[a])
            
            self.B_comb += b[self.disjoint_n_features:,:] @ b[self.disjoint_n_features:,:].T
            self.y_comb += reward * b[self.disjoint_n_features:,:]
            self.B_inv_comb = np.linalg.inv(self.B_comb)
        else:
            b = b.reshape((self.n_features, 1))

            self.B[a] += b @ b.T
            self.y[a] += reward * b
            self.B_inv[a] = np.linalg.inv(self.B[a])
        
        
class LinearContextualThompsonSampling:
    """
    Linear Contextual Thompson sampling algorithm implementation
    """

    def __init__(self, n_user_features, n_item_features, v, context="user", update_option = False):
        """
        Parameters
        ----------
        n_user_features : number
            dimension of user context
        n_item_features : number
            dimension of item context
        v : number
            LinearContextualThompsonSampling hyper-parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        update_option : boolean
            True : update every time
            False : update when offered item and offering item.
        """
        self.update_option = update_option
        if context == "user":
            self.context = 1
            self.n_features = n_user_features
        elif context == "both":
            self.context = 2
            self.n_features = n_user_features + n_item_features
        elif context == "user_interaction":
            self.context = 3
            #self.n_features = n_user_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + (n_user_features)* n_item_features
        elif context == "full":
            self.context = 4
            #self.n_features = n_user_features + n_item_features + (n_user_features - 1)* n_item_features
            self.n_features = n_user_features + n_item_features + (n_user_features)* n_item_features

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.v = v
        #self.algorithm = "LinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option)
        self.algorithm = "LinTS" + "_context_" + context + '_update' + str(update_option)
        
        
        

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        var = (self.v ** 2) * self.B_inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var)
        
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        elif self.context == 3:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))
        
        p = b_T @ mu_tilde
        return np.argmax(p)


    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        offered_item_features = pool_item_features[pool_offered]
        a = pool_idx[pool_offered]  # displayed article's index
        if self.context == 1:
            b = np.array(user)
        elif self.context == 2:
            b = np.hstack((user, offered_item_features))
        elif self.context == 3:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, interaction))
            b = b.reshape(-1)
        else:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((user, offered_item_features, interaction))
            b = b.reshape(-1)

        b = b.reshape((self.n_features, 1))

        self.B += b @ b.T
        self.y += reward * b
        self.B_inv = np.linalg.inv(self.B)
        
        
        
class ThompsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self, n_arms, update_option = False):
        """
        Parameters
        ----------
        n_arms : number
            total number of arms
        """
        self.update_option = update_option
        self.algorithm = "TS" + '_update' + str(update_option)
        self.n_arms = n_arms
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[pool_offered]  # displayed article's index

        self.alpha[a] += reward
        self.beta[a] += 1 - reward


class Ucb1:
    """
    UCB 1 algorithm implementation
    """

    def __init__(self, alpha, n_arms, update_option = False):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        n_arms : number
            total number of arms
        """
        self.update_option = update_option
        self.alpha = round(alpha, 1)
        self.n_arms = n_arms
        self.algorithm = "UCB1_alpha" + str(self.alpha) + '_update' + str(update_option)

        self.q = np.zeros(self.n_arms)  # average reward for each arm
        self.n = np.ones(self.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[pool_offered]  # displayed article's index

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, epsilon, n_arms, update_option = False):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        n_arms : number
            total number of arms
        """
        self.update_option = update_option
        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy
        self.n_arms = n_arms
        self.algorithm = "Egreedy_eps" + str(self.e) + '_update' + str(update_option)
        self.q = np.zeros(self.n_arms)  # average reward for each arm
        self.n = np.zeros(self.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx, pool_item_features = 0.):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, pool_offered, reward, user, pool_idx, pool_item_features = 0.):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[pool_offered]  # displayed article's index

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
