import numpy as np


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

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.v = v
        self.algorithm = "SemiparamLinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option)
        
        
        

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
        else:
            b_T = np.hstack((user, pool_item_features))
        
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
        else:
            b_T = np.hstack((user, pool_item_features))

        p = b_T @ mu_tilde.T
        ac_mc = list(np.argmax(p,axis = 0))
        pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(len(pool_idx))])
        b_mean=np.dot(b_T.T,pi_est)

        self.B += 2* (b_T[pool_offered,:] - b_mean).reshape(-1,1) @ (b_T[pool_offered,:] - b_mean).reshape(-1,1).T
        self.B += 2* ((b_T.T @ np.diag(pi_est)) @ b_T) - 2*(b_mean.reshape(-1,1) @ b_mean.reshape(-1,1).T)
        self.y += 2* reward* (b_T[pool_offered,:] - b_mean).reshape(-1,1)

        self.B_inv = np.linalg.inv(self.B)





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

        self.A = np.array([np.identity(self.n_features)] * self.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * self.n_arms)
        self.b = np.zeros((self.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "DisjointLinUCB_alpha"+str(self.alpha)+"_context_" + context + '_update' + str(update_option)

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
        else:
            x = np.hstack((user, pool_item_features))

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
        else:
            x = np.hstack((user, offered_item_features))

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

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.alpha = alpha
        self.algorithm = "LinUCB_alpha" + str(self.alpha) + "_context_" + context + '_update' + str(update_option)

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
        else:
            b_T = np.hstack((user, pool_item_features))
        
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
        else:
            b = np.hstack((user, offered_item_features))

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

        self.B = np.array([np.identity(self.n_features)] * self.n_arms)
        self.B_inv = np.array([np.identity(self.n_features)] * self.n_arms)
        self.y = np.zeros((self.n_arms, self.n_features, 1))
        self.v = v
        self.algorithm = "DisjointLinTS_v"+str(self.v)+"_context_" + context + '_update' + str(update_option)
        
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
        B_inv = self.B_inv[pool_idx]
        y = self.y[pool_idx]
        mu_tilde = []
        for idx in range(len(pool_idx)):
            mu_hat = (B_inv[idx,:,:] @ y[idx,:,:]).reshape(-1)
            var = (self.v ** 2) * B_inv[idx,:,:]
            mu_tilde.append(np.random.multivariate_normal(mu_hat,var))
        mu_tilde = np.vstack(mu_tilde).reshape(n_pool, self.n_features, 1)

        user = np.array([user] * n_pool)
        if self.context == 1:
            b_T = user
        else:
            b_T = np.hstack((user, pool_item_features))
        b_T = b_T.reshape(n_pool, self.n_features, 1)
        
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
        else:
            b = np.hstack((user, offered_item_features))

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

        self.B = np.array(np.identity(self.n_features))
        self.B_inv = np.array(np.identity(self.n_features))
        self.y = np.zeros((self.n_features, 1))
        self.v = v
        self.algorithm = "LinTS_v" + str(self.v) + "_context_" + context + '_update' + str(update_option)
        
        
        

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
        else:
            b_T = np.hstack((user, pool_item_features))
        
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
        else:
            b = np.hstack((user, offered_item_features))

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
