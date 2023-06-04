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
        self.algorithm = "SemiparamLinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option) + '_disjoint' + str(disjoint_option)
        
        
        

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
