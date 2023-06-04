import numpy as np


class SemiTS_hjs():
    def __init__(self, USER_NUM, ARM_NUM, DIMENSION, DELTA, OUR_GRAPH, OUR_LAPL,CONST_R, CONST_LAMBDA, CONST_V, TIME_HORIZON,add_intercept):      
        self.user_num = USER_NUM
        self.arm_num = ARM_NUM
        self.dim = DIMENSION
        self.delta = DELTA
        self.our_graph = OUR_GRAPH
        self.our_lapl = OUR_LAPL
        self.R = CONST_R
        self.T = TIME_HORIZON
        self.Lambda = CONST_LAMBDA
        self.v = CONST_V
        self.adjust_nu=True
        #if self.v == 0:
        #    self.v = 2 * self.R * np.sqrt( 9 * self.dim * (np.log(self.T / self.Lambda) + 1 / self.Lambda) )
        self.users_B_list = [np.identity(self.dim)*self.Lambda for i in range(0, self.user_num)]
        self.users_B_inv_list = [np.identity(self.dim)/self.Lambda for i in range(0, self.user_num)]
        self.users_y_list = [np.zeros(self.dim) for i in range(0, self.user_num)]
       

        self.users_B2_list = [np.identity(self.dim+1)*self.Lambda for i in range(0, self.user_num)]
        self.users_B2_inv_list = [np.identity(self.dim+1)/self.Lambda for i in range(0, self.user_num)]
        self.users_y2_list = [np.zeros(self.dim+1) for i in range(0, self.user_num)]
        self.intercept_list=[0 for i in range(0,self.user_num)]
   
   
    def choose_arm(self, SELECTED_USER, arms_features):
        self.N=len(arms_features)
        B_INV=self.users_B_inv_list[SELECTED_USER]
       
        self.mu_hat = np.matmul(B_INV, self.users_y_list[SELECTED_USER])
       
        self.V=(self.v ** 2) * B_INV
        mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)
        
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(self.N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        self.SELECTED_USER=SELECTED_USER
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        
        mu_hat = (self.B_inv @ self.y).reshape(-1)
        var = (self.v ** 2) * self.B_inv
        mu_tilde=np.random.multivariate_normal(mu_hat,var,1000)

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        elif self.context == 2:
            b_T = np.hstack((user, pool_item_features))
        else:
            interaction = np.einsum('ij,ik->ijk',user[:,:(user.shape[1]-1)],pool_item_features).reshape(pool_item_features.shape[0],-1)
            b_T = np.hstack((user, pool_item_features, interaction))

        p = b_T @ mu_tilde.T
        ac_mc = list(np.argmax(p,axis = 0))
        pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
        b_mean=np.dot(b_T.T,pi_est)

        # Fix 20220412
        #self.B += (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
        #self.B += ((b_T.T @ np.diag(pi_est)) @ b_T) - (b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        #self.y += 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)
        self.B += 2*((b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T)
        self.B += 2* ((b_T.T @ np.diag(pi_est)) @ b_T) - 2*(b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        self.y += 4* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

        self.B_inv = np.linalg.inv(self.B)
        
        
        
        mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
        est_mc=list((np.dot(self.arms_features,mu_mc.T)).T)
        ac_mc=list(np.argmax(est_mc,axis=1))
        pi_est=np.array([float(ac_mc.count(i))/len(ac_mc) for i in range(self.N)])
        b_mean=np.dot(np.transpose(np.array(self.arms_features)),pi_est)
       
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.outer(self.chosen_arm_feature-b_mean,
                                                                                                 self.chosen_arm_feature-b_mean)
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.dot(np.dot(np.transpose(self.arms_features),np.diag(pi_est)),self.arms_features)-np.outer(b_mean,b_mean)
       
        temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
        temp_armf=np.copy(self.chosen_arm_feature-b_mean)
        self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
        for i in range(self.N):
            temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
            temp_armf=np.sqrt(pi_est[i])*(self.arms_features[i]-b_mean)
            self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
       
        if self.adjust_nu:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(-self.intercept_list[self.SELECTED_USER]+OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
        else:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)

        if self.adjust_nu:
            newf=np.array([1.]+list(self.chosen_arm_feature))
            self.users_B2_list[self.SELECTED_USER] = self.users_B2_list[self.SELECTED_USER] + np.outer(newf,newf)
            temp_B2inv=np.copy(self.users_B2_inv_list[self.SELECTED_USER])
            self.users_B2_inv_list[self.SELECTED_USER]=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
            self.users_y2_list[self.SELECTED_USER] =  self.users_y2_list[self.SELECTED_USER] + OBSERVED_REWARD * newf
            self.intercept_list[self.SELECTED_USER] = np.matmul(self.users_B2_inv_list[self.SELECTED_USER], self.users_y2_list[self.SELECTED_USER])[0]
            

            
class SemiTS_Single_js():
    def __init__(self, ARM_NUM, DIMENSION, CONST_LAMBDA, CONST_V):       
        self.arm_num = ARM_NUM
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
        
    
    def choose_arm(self, SELECTED_USER, arms_features):
        N=len(arms_features)
        B_INV=self.B_inv
        self.V=(self.v ** 2) * B_INV
        mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)     
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        #self.SELECTED_USER=SELECTED_USER
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        N=len(self.arms_features)
        mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
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
            
            
class SemiTS_Single():
    def __init__(self, USER_NUM, ARM_NUM, DIMENSION, DELTA, OUR_GRAPH, OUR_LAPL,CONST_R, CONST_LAMBDA, CONST_V, TIME_HORIZON,add_intercept):       
        self.user_num = USER_NUM
        self.arm_num = ARM_NUM
        self.dim = DIMENSION
        self.delta = DELTA
        self.our_graph = OUR_GRAPH
        self.our_lapl = OUR_LAPL
        self.R = CONST_R
        self.T = TIME_HORIZON
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
        
    
    def choose_arm(self, SELECTED_USER, arms_features):
        N=len(arms_features)
        B_INV=self.B_inv
        self.V=(self.v ** 2) * B_INV
        mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)     
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        #self.SELECTED_USER=SELECTED_USER
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        N=len(self.arms_features)
        mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
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

class SemiTS():
    def __init__(self, USER_NUM, ARM_NUM, DIMENSION, DELTA, OUR_GRAPH, OUR_LAPL,CONST_R, CONST_LAMBDA, CONST_V, TIME_HORIZON,add_intercept):      
        self.user_num = USER_NUM
        self.arm_num = ARM_NUM
        self.dim = DIMENSION
        self.delta = DELTA
        self.our_graph = OUR_GRAPH
        self.our_lapl = OUR_LAPL
        self.R = CONST_R
        self.T = TIME_HORIZON
        self.Lambda = CONST_LAMBDA
        self.v = CONST_V
        self.adjust_nu=True
        #if self.v == 0:
        #    self.v = 2 * self.R * np.sqrt( 9 * self.dim * (np.log(self.T / self.Lambda) + 1 / self.Lambda) )
        self.users_B_list = [np.identity(self.dim)*self.Lambda for i in range(0, self.user_num)]
        self.users_B_inv_list = [np.identity(self.dim)/self.Lambda for i in range(0, self.user_num)]
        self.users_y_list = [np.zeros(self.dim) for i in range(0, self.user_num)]
       

        self.users_B2_list = [np.identity(self.dim+1)*self.Lambda for i in range(0, self.user_num)]
        self.users_B2_inv_list = [np.identity(self.dim+1)/self.Lambda for i in range(0, self.user_num)]
        self.users_y2_list = [np.zeros(self.dim+1) for i in range(0, self.user_num)]
        self.intercept_list=[0 for i in range(0,self.user_num)]
   
   
    def choose_arm(self, SELECTED_USER, arms_features):
        self.N=len(arms_features)
        B_INV=self.users_B_inv_list[SELECTED_USER]
       
        self.mu_hat = np.matmul(B_INV, self.users_y_list[SELECTED_USER])
       
        self.V=(self.v ** 2) * B_INV
        mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)    
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(self.N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        self.SELECTED_USER=SELECTED_USER
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
        est_mc=list((np.dot(self.arms_features,mu_mc.T)).T)
        ac_mc=list(np.argmax(est_mc,axis=1))
        pi_est=np.array([float(ac_mc.count(i))/len(ac_mc) for i in range(self.N)])
        b_mean=np.dot(np.transpose(np.array(self.arms_features)),pi_est)
       
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.outer(self.chosen_arm_feature-b_mean,
                                                                                                 self.chosen_arm_feature-b_mean)
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.dot(np.dot(np.transpose(self.arms_features),np.diag(pi_est)),self.arms_features)-np.outer(b_mean,b_mean)
       
        temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
        temp_armf=np.copy(self.chosen_arm_feature-b_mean)
        self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
        for i in range(self.N):
            temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
            temp_armf=np.sqrt(pi_est[i])*(self.arms_features[i]-b_mean)
            self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
       
        if self.adjust_nu:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(-self.intercept_list[self.SELECTED_USER]+OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
        else:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)

        if self.adjust_nu:
            newf=np.array([1.]+list(self.chosen_arm_feature))
            self.users_B2_list[self.SELECTED_USER] = self.users_B2_list[self.SELECTED_USER] + np.outer(newf,newf)
            temp_B2inv=np.copy(self.users_B2_inv_list[self.SELECTED_USER])
            self.users_B2_inv_list[self.SELECTED_USER]=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
            self.users_y2_list[self.SELECTED_USER] =  self.users_y2_list[self.SELECTED_USER] + OBSERVED_REWARD * newf
            self.intercept_list[self.SELECTED_USER] = np.matmul(self.users_B2_inv_list[self.SELECTED_USER], self.users_y2_list[self.SELECTED_USER])[0]




class SemiRGraphTS():
    def __init__(self, USER_NUM, ARM_NUM, DIMENSION, DELTA, OUR_GRAPH, OUR_LAPL,CONST_R, CONST_LAMBDA, CONST_V, TIME_HORIZON,add_intercept):      
        self.user_num = USER_NUM
        self.arm_num = ARM_NUM
        self.dim = DIMENSION
        self.delta = DELTA
        self.our_graph = OUR_GRAPH
        self.our_lapl = OUR_LAPL
        self.R = CONST_R
        self.T = TIME_HORIZON
        self.Lambda = CONST_LAMBDA
        self.v = CONST_V
        self.adjust_nu=True
       
        #if self.v == 0:
        #    self.v = 2 * self.R * np.sqrt( 9 * self.dim * (np.log(self.T / self.Lambda) + 1 / self.Lambda) )
        self.users_B_list = [self.Lambda * self.our_lapl[i, i] * np.identity(self.dim) for i in range(0, self.user_num)]
        self.users_B_inv_list = [(1/self.Lambda) * (1/self.our_lapl[i, i]) * np.identity(self.dim) for i in range(0, self.user_num)]
        self.users_y_list = [np.zeros(self.dim) for i in range(0, self.user_num)]
        self.users_mu_bar_list = [np.zeros(self.dim) for i in range(0, self.user_num)]
       
       
        self.users_B2_list = [self.Lambda * self.our_lapl[i, i] * np.identity(self.dim+1) for i in range(0, self.user_num)]
        self.users_B2_inv_list = [(1/self.Lambda) * (1/self.our_lapl[i, i]) * np.identity(self.dim+1) for i in range(0, self.user_num)]
        self.users_y2_list = [np.zeros(self.dim+1) for i in range(0, self.user_num)]
        self.intercept_list=[0 for i in range(0,self.user_num)]

   
    def choose_arm(self, SELECTED_USER, arms_features):
        self.N=len(arms_features)
        # mu_hat
        B_INV=self.users_B_inv_list[SELECTED_USER]
        temp = 0
        for i in range(0, self.user_num):
            temp += self.our_lapl[SELECTED_USER, i] * self.users_mu_bar_list[i]
        temp -= self.our_lapl[SELECTED_USER, SELECTED_USER] * self.users_mu_bar_list[SELECTED_USER]
       
        self.mu_hat = self.users_mu_bar_list[SELECTED_USER] - self.Lambda * np.matmul(B_INV, temp)  
       
        # mu_tilde
        # construct covariance matrix
        B_CURR = self.users_B_list[SELECTED_USER] + self.Lambda * self.our_lapl[i, i] * np.identity(self.dim)
        for i in range(0, self.user_num):
            if (i != SELECTED_USER):
                B_CURR += ((self.Lambda * self.our_lapl[SELECTED_USER, i])**2) * self.users_B_inv_list[i]
        B_CURR_INV = np.linalg.pinv(B_CURR)
       
        self.V=(self.v ** 2) * B_CURR_INV
        mu_tilde= np.random.multivariate_normal(mean = self.mu_hat, cov = self.V)    
        est_reward=[np.dot(arms_features[i],mu_tilde) for i in range(self.N)]
        chosen_arm = est_reward.index(max(est_reward))
        self.chosen_arm_feature = arms_features[chosen_arm]
        self.SELECTED_USER=SELECTED_USER
        self.arms_features=arms_features
        return chosen_arm

    def update(self, OBSERVED_REWARD):
        mu_mc=np.random.multivariate_normal(self.mu_hat,self.V,1000)
        est_mc=list((np.dot(self.arms_features,mu_mc.T)).T)
        ac_mc=list(np.argmax(est_mc,axis=1))
        pi_est=np.array([float(ac_mc.count(i))/len(ac_mc) for i in range(self.N)])
        b_mean=np.dot(np.transpose(np.array(self.arms_features)),pi_est)
       
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.outer(self.chosen_arm_feature-b_mean,
                                                                                                 self.chosen_arm_feature-b_mean)
        self.users_B_list[self.SELECTED_USER] = self.users_B_list[self.SELECTED_USER] + np.dot(np.dot(np.transpose(self.arms_features),np.diag(pi_est)),self.arms_features)-np.outer(b_mean,b_mean)
       
        temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
        temp_armf=np.copy(self.chosen_arm_feature-b_mean)
        self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
        for i in range(self.N):
            temp_Binv=np.copy(self.users_B_inv_list[self.SELECTED_USER])
            temp_armf=np.sqrt(pi_est[i])*(self.arms_features[i]-b_mean)
            self.users_B_inv_list[self.SELECTED_USER]=temp_Binv-((temp_Binv).dot(np.outer(temp_armf,temp_armf))).dot(temp_Binv)/(1.+np.dot(temp_armf,(temp_Binv).dot(temp_armf)))
       
        if self.adjust_nu:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(-self.intercept_list[self.SELECTED_USER]+OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
        else:
            self.users_y_list[self.SELECTED_USER] =  self.users_y_list[self.SELECTED_USER] + 2*(OBSERVED_REWARD) * (self.chosen_arm_feature-b_mean)
       
        self.users_mu_bar_list[self.SELECTED_USER] = np.matmul(self.users_B_inv_list[self.SELECTED_USER], self.users_y_list[self.SELECTED_USER])
       
        if self.adjust_nu:
            newf=np.array([1.]+list(self.chosen_arm_feature))
            self.users_B2_list[self.SELECTED_USER] = self.users_B2_list[self.SELECTED_USER] + np.outer(newf,newf)
            temp_B2inv=np.copy(self.users_B2_inv_list[self.SELECTED_USER])
            self.users_B2_inv_list[self.SELECTED_USER]=temp_B2inv-((temp_B2inv).dot(np.outer(newf,newf))).dot(temp_B2inv)/(1.+np.dot(newf,(temp_B2inv).dot(newf)))
            self.users_y2_list[self.SELECTED_USER] =  self.users_y2_list[self.SELECTED_USER] + OBSERVED_REWARD * newf
            self.intercept_list[self.SELECTED_USER] = np.matmul(self.users_B2_inv_list[self.SELECTED_USER], self.users_y2_list[self.SELECTED_USER])[0]