import numpy as np


def makedata(n_users, n_items, R, T):
    errors=list()
    user_idx = np.random.choice(np.arange(n_users), T)
    for t in range(T):
        errors.append(np.random.multivariate_normal(np.zeros(n_items),np.eye(n_items)*(R**2)))
    return([user_idx,errors])



def cumul_opt(T, p_min, p_max, Data, initial_model, change_model, change_point, user_features, item_features):
    reward = np.zeros(T)
    vs = np.zeros(T)
    for t in range(T):
        user = user_features[Data[0][t],:].reshape(1,-1)
        example_x = []
        for i in range(item_features.shape[0]):
            pool_item_features = np.array(item_features[i,:]).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            x = np.hstack((user, pool_item_features, interaction))
            example_x.append(x)
        example_x = np.concatenate(example_x, axis = 0)
        if t < change_point:
            tru_nonzero = initial_model.predict(example_x)
        else:
            tru_nonzero = change_model.predict(example_x)
        opt_nonzero = np.argmax(tru_nonzero)
        if tru_nonzero[opt_nonzero]<0:
            pt_opt=p_min
        else:
            pt_opt=p_max            
        reward[t]=tru_nonzero[opt_nonzero]*pt_opt
        vs[t]=np.log(t+1)
    return([np.cumsum(reward),vs])


def cumul_model(T, Data, vs, initial_model, change_model, change_point, mab_model, user_features, item_features):
    n_users = user_features.shape[0]
    n_items = item_features.shape[0]

    reward = np.zeros(T)
    pool_idx = np.arange(n_items)
    
    for t in range(T):
        #user = user_features[Data[0][t],:].reshape(1,-1)
        user = user_features[Data[0][t],:]
        errors = Data[1][t]
        pool_item_features = item_features[pool_idx,:]
        chosen = mab_model.choose_arm(t, user, pool_idx, pool_item_features)
        
        chosen_item_features = np.array(item_features[chosen,:]).reshape(1,-1)
        interaction = np.einsum('ij,ik->ijk',user.reshape(1,-1),chosen_item_features).reshape(chosen_item_features.shape[0],-1)
        x = np.hstack((user.reshape(1,-1), chosen_item_features, interaction))
        if t < change_point:
            reward[t] = initial_model.predict(x)
        else:
            reward[t] = change_model.predict(x)
        mab_model.update(chosen, reward[t] + errors[chosen] + vs[t], user, pool_idx, pool_item_features)
    return(np.cumsum(reward))

def cumul_opt_v2(T, p_min, p_max, Data, model_list, user_features, item_features):
    total_model_num = len(model_list)
    change_num = T/total_model_num
    reward = np.zeros(T)
    vs = np.zeros(T)
    for t in range(T):
        user = user_features[Data[0][t],:].reshape(1,-1)
        example_x = []
        for i in range(item_features.shape[0]):
            pool_item_features = np.array(item_features[i,:]).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            x = np.hstack((user, pool_item_features, interaction))
            example_x.append(x)
        example_x = np.concatenate(example_x, axis = 0)
        
        model_idx = int(t/change_num)
        tru_nonzero = model_list[model_idx].predict(example_x)
        '''
        if t < change_point:
            tru_nonzero = initial_model.predict(example_x)
        else:
            tru_nonzero = change_model.predict(example_x)
        '''
        opt_nonzero = np.argmax(tru_nonzero)
        if tru_nonzero[opt_nonzero]<0:
            pt_opt=p_min
        else:
            pt_opt=p_max            
        reward[t]=tru_nonzero[opt_nonzero]*pt_opt
        vs[t]=np.log(t+1)
    return([np.cumsum(reward),vs])


def cumul_model_v2(T, Data, vs, model_list, mab_model, user_features, item_features):
    total_model_num = len(model_list)
    change_num = T/total_model_num
    n_users = user_features.shape[0]
    n_items = item_features.shape[0]

    reward = np.zeros(T)
    pool_idx = np.arange(n_items)
    
    for t in range(T):
        #user = user_features[Data[0][t],:].reshape(1,-1)
        user = user_features[Data[0][t],:]
        errors = Data[1][t]
        pool_item_features = item_features[pool_idx,:]
        chosen = mab_model.choose_arm(t, user, pool_idx, pool_item_features)
        
        chosen_item_features = np.array(item_features[chosen,:]).reshape(1,-1)
        interaction = np.einsum('ij,ik->ijk',user.reshape(1,-1),chosen_item_features).reshape(chosen_item_features.shape[0],-1)
        x = np.hstack((user.reshape(1,-1), chosen_item_features, interaction))
        
        model_idx = int(t/change_num)
        reward[t] = model_list[model_idx].predict(x)
        '''
        if t < change_point:
            reward[t] = initial_model.predict(x)
        else:
            reward[t] = change_model.predict(x)
        '''
        mab_model.update(chosen, reward[t] + errors[chosen] + vs[t], user, pool_idx, pool_item_features)
    return(np.cumsum(reward))



def cumul_opt_v3(T, p_min, p_max, Data, model_list, user_features, item_features, vs_option):
    total_model_num = len(model_list)
    change_num = T/total_model_num
    reward = np.zeros(T)
    vs = np.zeros(T)
    for t in range(T):
        user = user_features[Data[0][t],:].reshape(1,-1)
        example_x = []
        for i in range(item_features.shape[0]):
            pool_item_features = np.array(item_features[i,:]).reshape(1,-1)
            interaction = np.einsum('ij,ik->ijk',user,pool_item_features).reshape(pool_item_features.shape[0],-1)
            x = np.hstack((user, pool_item_features, interaction))
            example_x.append(x)
        example_x = np.concatenate(example_x, axis = 0)
        
        model_idx = int(t/change_num)
        tru_nonzero = model_list[model_idx].predict(example_x)
        '''
        if t < change_point:
            tru_nonzero = initial_model.predict(example_x)
        else:
            tru_nonzero = change_model.predict(example_x)
        '''
        opt_nonzero = np.argmax(tru_nonzero)
        if tru_nonzero[opt_nonzero]<0:
            pt_opt=p_min
        else:
            pt_opt=p_max            
        reward[t]=tru_nonzero[opt_nonzero]*pt_opt
        if vs_option == 0:
            vs[t]=0 ## Case (i): nu(t)=0
        elif vs_option == 1:
            vs[t]=-reward[t] ## Case (ii): nu(t)=-b_{a^*(t)}^T mu
        elif vs_option == 2:
            vs[t]=np.log(t+1)
        elif vs_option == 3:
            vs[t]=np.cos(t)
        elif vs_option == 4:
            vs[t]=np.cos((t*np.pi)/(T/2))
        elif vs_option == 5:
            vs[t]=np.cos((t*np.pi)/(T/2))*np.log(t+1)
        elif vs_option == 6:
            vs[t]=np.cos((t*np.pi)/(T/2))*(-reward[t])
    return([np.cumsum(reward),vs])