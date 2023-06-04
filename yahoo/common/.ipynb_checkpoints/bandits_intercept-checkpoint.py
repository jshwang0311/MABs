import numpy as np


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
        self.n_features += 1

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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            b_T = np.hstack((intercept, user))
        elif self.context == 2:
            b_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5) :
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, pool_item_features, interaction))
        
        if self.context == 5:
            B_inv_disjoint = self.B_inv_disjoint[pool_idx]
            y_disjoint = self.y_disjoint[pool_idx]
            mu_tilde_disjoint = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv_disjoint[idx,:,:] @ y_disjoint[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv_disjoint[idx,:,:]
                try:
                    mu_tilde_disjoint.append(np.random.multivariate_normal(mu_hat,var))
                except:
                    mu_tilde_disjoint.append(np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky'))
            #mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features, 1)
            mu_tilde_disjoint = np.vstack(mu_tilde_disjoint).reshape(n_pool, self.disjoint_n_features)

            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            try:
                mu_tilde_comb=np.random.multivariate_normal(mu_hat,var)
            except:
                mu_tilde_comb = np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky')
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
                    try:
                        mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
                    except:
                        mu_tilde.append(np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky'))
                mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)


                b_T = b_T.reshape(n_pool, self.n_features, 1)
                p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T

            else:
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                var = (self.v ** 2) * self.B_inv
                try:
                    mu_tilde=np.random.multivariate_normal(mu_hat,var)
                except : 
                    mu_tilde = np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky')
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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            b_T = np.hstack((intercept, user))
        elif self.context == 2:
            b_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, pool_item_features, interaction))
            
        
        if self.context == 5:
            offered_item_features = pool_item_features[pool_offered]
            a = pool_idx[pool_offered]  # displayed article's index
            ###
            mu_hat = (self.B_inv_disjoint[a,:,:] @ self.y_disjoint[a,:,:]).reshape(-1)
            var = (self.v ** 2) * self.B_inv_disjoint[a,:,:]
            try:
                mu_tilde_disjoint=np.random.multivariate_normal(mu_hat,var,1000)
            except:
                mu_tilde_disjoint=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            try:
                mu_tilde_comb=np.random.multivariate_normal(mu_hat,var,1000)
            except:
                mu_tilde_comb=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')

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
                try:
                    mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                except:
                    mu_tilde=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')
                    

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
        self.n_features += 1

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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            a_T = np.hstack((intercept, user))
        elif self.context == 2:
            a_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((intercept, user, pool_item_features, interaction))
        
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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            a_T = np.hstack((intercept, user))
        elif self.context == 2:
            a_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            a_T = np.hstack((intercept, user, pool_item_features, interaction))
            
        
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
        self.n_features += 1

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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            b_T = np.hstack((intercept, user))
        elif self.context == 2:
            b_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, pool_item_features, interaction))
            
            
        if self.context == 5:
            B_inv = self.B_inv_disjoint[pool_idx]
            B2 = self.B2_disjoint[pool_idx]
            y = self.y_disjoint[pool_idx]
            mu_tilde_disjoint = []
            for idx in range(len(pool_idx)):
                mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
                var = (self.v ** 2) * B_inv[idx,:,:] @ B2[idx,:,:] @ B_inv[idx,:,:]
                try:
                    mu_tilde_disjoint.append(np.random.multivariate_normal(mu_hat,var))
                except:
                    mu_tilde_disjoint.append(np.random.default_rng().multivariate_normal(mu_hat,var, method = 'cholesky'))
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
                    try:
                        mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
                    except:
                        mu_tilde.append(np.random.default_rng().multivariate_normal(mu_hat,var, method = 'cholesky'))
                mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)
                b_T = b_T.reshape(n_pool, self.n_features, 1)

                p = np.transpose(mu_tilde, (0, 2, 1)) @ b_T
            else:
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                inv  = self.B_inv @ self.B2 @ self.B_inv
                var = (self.v ** 2) * inv
                try:
                    mu_tilde=np.random.multivariate_normal(mu_hat,var)
                except:
                    mu_tilde=np.random.default_rng().multivariate_normal(mu_hat,var, method = 'cholesky')

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
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            b_T = np.hstack((intercept, user))
        elif self.context == 2:
            b_T = np.hstack((intercept, user, pool_item_features))
        elif (self.context == 3 or self.context == 5):
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, pool_item_features, interaction))
        
        
        
        if self.context == 5:
            offered_item_features = pool_item_features[pool_offered]
            a = pool_idx[pool_offered]  # displayed article's index
            ###
            mu_hat = (self.B_inv_disjoint[a,:,:] @ self.y_disjoint[a,:,:]).reshape(-1)
            var = (self.v ** 2) * self.B_inv_disjoint[a,:,:]
            try:
                mu_tilde_disjoint=np.random.multivariate_normal(mu_hat,var,1000)
            except:
                mu_tilde_disjoint=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')
            
            mu_hat = (self.B_inv_comb @ self.y_comb).reshape(-1)
            var = (self.v ** 2) * self.B_inv_comb
            try:
                mu_tilde_comb=np.random.multivariate_normal(mu_hat,var,1000)
            except:
                mu_tilde_comb=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')

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
                try:
                    mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                except:
                    mu_tilde=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')
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
                mu_hat = (self.B_inv @ self.y).reshape(-1)
                inv  = self.B_inv @ self.B2 @ self.B_inv
                var = (self.v ** 2) * inv
                try:
                    mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)
                except:
                    mu_tilde=np.random.default_rng().multivariate_normal(mu_hat,var, 1000, method = 'cholesky')

                p = b_T @ mu_tilde.T
                ac_mc = list(np.argmax(p,axis = 0))
                pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
                b_mean=np.dot(b_T.T,pi_est)


                self.B = (1-self.gamma) * self.lbd * np.array(np.identity(self.n_features)) + self.gamma* self.B + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                self.B_inv = np.linalg.inv(self.B)

                self.B2 = (1-self.gamma**2) * self.lbd * np.array(np.identity(self.n_features)) + (self.gamma**2) * self.B2 + (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
                self.B2 += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
                
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
        
        self.n_features += 1

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
        try:
            mu_tilde=np.random.multivariate_normal(mu_hat,var)
        except:
            mu_tilde = np.random.default_rng().multivariate_normal(mu_hat,var,method = 'cholesky')
        
        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        intercept = np.repeat(1.,n_pool).reshape(n_pool,1)
        if self.context == 1:
            b_T = np.hstack((intercept, user))
        elif self.context == 2:
            b_T = np.hstack((intercept, user, pool_item_features))
        elif self.context == 3:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, interaction))
        else:
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((intercept, user, pool_item_features, interaction))
        
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
        intercept = np.array(1.).reshape(1,-1)
        if self.context == 1:
            b = np.hstack((intercept, user))
        elif self.context == 2:
            b = np.hstack((intercept, user, offered_item_features))
        elif self.context == 3:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((intercept, user, interaction))
            b = b.reshape(-1)
        else:
            user = np.array(user).reshape(1,-1)
            offered_item_features = np.array(offered_item_features).reshape(1,-1)
            #interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],offered_item_features).reshape(offered_item_features.shape[0],-1)
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1])],offered_item_features).reshape(offered_item_features.shape[0],-1)
            b = np.hstack((intercept, user, offered_item_features, interaction))
            b = b.reshape(-1)

        b = b.reshape((self.n_features, 1))

        self.B += b @ b.T
        self.y += reward * b
        self.B_inv = np.linalg.inv(self.B)