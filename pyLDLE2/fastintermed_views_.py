import pdb
import time
import numpy as np
import copy

from .util_ import print_log, compute_zeta, procrustes_cost
from .util_ import Param

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

import multiprocess as mp
from multiprocess import shared_memory
import itertools

import queue
import copy
import torch

# merging s to m
def merging_cost(s, m, Utilde_s, Utilde_m, d_e, local_param, intermed_opts):
    if intermed_opts['cost_fn'] == 'distortion':
        Utilde_s_U_Utilde_m = list(Utilde_s.union(Utilde_m))
        c = compute_zeta(d_e[np.ix_(Utilde_s_U_Utilde_m,Utilde_s_U_Utilde_m)],
                         local_param.eval_({'view_index': m,
                                            'data_mask': Utilde_s_U_Utilde_m}))
    else: # procrustes
        U_s = list(Utilde_s)
        c = procrustes_cost(local_param.eval_({'view_index': s, 'data_mask': U_s}),
                            local_param.eval_({'view_index': m, 'data_mask': U_s}))

    return c


# Computes cost_k, d_k (dest_k)
def batch_cost_of_moving(U,U_, local_param, c, n_C,
                   Utilde, eta_min, eta_max, intermed_opts):
    """
        c:  cluster assignments
        n_c: size of each cluster
    """
    #TODO
    # We first list all faisable (k,m) pairs:

    # For each of this pair, we compute the new U.
    # We then padd all the U in order to make the batched computation feasible.
    
    # Then we evaluate the cost of each of these movement.
    
    # Finally we reshape these cost and take the best movement for every k
    
          


    # Compute |C_{c_k}|
    n_C_c_k = n_C[c]

    
    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1
    
    # Compute neighboring clusters c_{U_k} of x_k

    c_U_k = c[U]

    c_U_k_uniq = np.unique(c_U_k).tolist()
    cost_x_k_to = np.zeros(len(c_U_k_uniq)) + np.inf

    U_k_list = list(U_)
    
    # Iterate over all m in c_{U_k}
    i = 0
    for m in c_U_k_uniq:
        if m == c_k:
            i += 1
            continue
            
        # Compute |C_{m}|
        n_C_m = n_C[m]
        # Check if |C_{m}| < eta_{max}. If not
        # then mth cluster has reached the max
        # allowed size of the cluster. Move on.
        if n_C_m >= eta_max:
            i += 1
            continue
        
        # Check if |C_{m}| >= |C_{c_k}|. If yes, then
        # mth cluster satisfies all required conditions
        # and is a candidate cluster to move x_k in.
        if n_C_m >= n_C_c_k:
            if intermed_opts['cost_fn'] == 'distortion':
                # Compute union of Utilde_m U_k
                U_k_U_Utilde_m = list(U_k.union(Utilde[m]))
                # Compute the cost of moving x_k to mth cluster,
                # that is cost_{x_k \rightarrow m}
                cost_x_k_to[i] = compute_zeta(d_e[np.ix_(U_k_U_Utilde_m,U_k_U_Utilde_m)],
                                      local_param.eval_({'view_index': m,
                                                         'data_mask': U_k_U_Utilde_m}))
            else: # procrustes
                cost_x_k_to[i] = procrustes_cost(local_param.eval_({'view_index': k, 'data_mask': U_k_list}),
                                                 local_param.eval_({'view_index': m, 'data_mask': U_k_list}))
                
        i += 1

    ### Batch version:
    S = 


    
    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
    else:
        dest_k = c_U_k_uniq[dest_k]
        
    return cost_k, dest_k

class FastIntermedViews:
    def __init__(self, exit_at, verbose=True, debug=False):
        self.exit_at = exit_at
        self.verbose = verbose
        self.debug = debug
        
        self.c = None
        self.C = None
        self.n_C = None
        self.Utilde = None
        self.intermed_param = None
        
        self.local_start_time = time.perf_counter()
        self.global_start_time = time.perf_counter()
    
    def log(self, s='', log_time=False):
        if self.verbose:
            self.local_start_time = print_log(s, log_time,
                                              self.local_start_time, 
                                              self.global_start_time)
    
    def best(self, d, U, local_param, intermed_opts):
        """
            U: before was the mask, now (N,k_nn) indices of neighbors
        """
        n = U.shape[0]
        c = torch.arange(n,device="cuda")
        n_C = torch.zeros(n,dtype=torch.float32,device="cuda") + 1
        Clstr = list(map(set, np.arange(n).reshape((n,1)).tolist()))
        
        U_ = U.clone()
        Utilde = U.clone()
        # indices = U.reshape(-1)
        # # indptr = U.indptr
        # Utilde = []
        # U_ = []
        # neigh_ind = []
        # for i in range(n):
        #     col_inds = indices[indptr[i]:indptr[i+1]]
        #     Utilde.append(set(col_inds))
        #     U_.append(set(col_inds))
        #     neigh_ind.append(col_inds)
        # neigh_ind = np.array(neigh_ind)


        eta_max = intermed_opts['eta_max']
        n_proc = intermed_opts['n_proc']
        
        cost = np.zeros(n)+np.inf
        dest = np.zeros(n,dtype='int')-1

        
        # Vary eta from 2 to eta_{min}
        self.log('Constructing intermediate views.')
        for eta in range(2,intermed_opts['eta_min']+1):
           
            
            # Sequential version of above
            cost, dest = batch_cost_of_moving( U, U_, local_param,
                                                  c, n_C, Utilde, eta, eta_max,intermed_opts)
            
            # Compute point with minimum cost
            # Compute k and cost^* 
            k = np.argmin(np_cost)
            cost_star = np_cost[k]
            
            self.log('Costs computed when eta = %d.' % eta, log_time=True)
            
            # Loop until minimum cost is inf
            total_len_S = 0
            ctr = 0
            while cost_star < np.inf:
                # Move x_k from cluster s to
                # dest_k and update variables
                s = c[k]
                dest_k = np_dest[k]
                c[k] = dest_k
                n_C[s] -= 1
                n_C[dest_k] += 1
                Clstr[s].remove(k)
                Clstr[dest_k].add(k)
                Utilde[dest_k] = U_[k].union(Utilde[dest_k])
                Utilde[s] = set(itertools.chain.from_iterable(neigh_ind[list(Clstr[s])]))
                
                # Compute the set of points S for which 
                # cost of moving needs to be recomputed
                if n_C[s] > 0:
                    S_ = (c==dest_k) | (np_dest==dest_k) | np.array(U[:,list(Clstr[s])].sum(1),dtype=bool).flatten()
                else:
                    S_ = (c==dest_k) | (np_dest==dest_k) | (np_dest==s)
                S = np.where(S_)[0].tolist()
                len_S = len(S)
                total_len_S += len_S
                ctr += 1
                
                ###########################################
                # cost, dest update for k in S
                ###########################################
                if False and len_S > intermed_opts['len_S_thresh']: # do parallel update (almost never true)
                    proc = []
                    n_proc_ = min(n_proc, len_S)
                    chunk_sz = int(len_S/n_proc_)
                    for p_num in range(n_proc_):
                        proc.append(mp.Process(target=target_proc,
                                               args=(p_num,chunk_sz,len_S,Utilde,n_C,c,S),
                                               daemon=True))
                        proc[-1].start()

                    for p_num in range(n_proc_):
                        proc[p_num].join()
                    ###########################################
                else: # sequential update
                    for k in S:
                        np_cost[k], np_dest[k] = cost_of_moving(k, d_e, neigh_ind[k], U_[k], local_param,
                                                                c, n_C, Utilde, eta, eta_max, intermed_opts)
                ###########################################
                ###########################################
                # Recompute point with minimum cost
                # Recompute k and cost^*
                k = np.argmin(np_cost)
                cost_star = np_cost[k]
            
            print('ctr=%d, total_len_S=%d, avg_len_S=%0.3f' % (ctr, total_len_S, total_len_S/(ctr+1e-12)))
            print('Remaining #nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            self.log('Done with eta = %d.' % eta, log_time=True)
        del U_
        del Clstr
        del Utilde
        del cost
        del np_cost
        shm_cost.close()
        shm_cost.unlink()
        del dest
        del np_dest
        shm_dest.close()
        shm_dest.unlink()
        return c, n_C
    
    def match_n_merge(self, d, d_e, U, neigh_ind, local_param, intermed_opts):
        n = d_e.shape[0]
        c = np.arange(n, dtype=int)
        n_C = np.ones(n, dtype=int)
        Utilde = []
        for k in range(n):
            neigh_ind_k = set(neigh_ind[k])
            Utilde.append(neigh_ind_k)
            
        n_proc = intermed_opts['n_proc']
        n_times = intermed_opts['n_times']
        
        # Compute merging costs between
        # each pair of nbring clusters
        def target_proc(p_num, chunk_sz, q_, clstrs, nbr_clstrs_of, clstr_dist):
            start_ind = p_num*chunk_sz
            if p_num == (n_proc-1):
                end_ind = len(clstrs)
            else:
                end_ind = (p_num+1)*chunk_sz
            n_rows = 0
            for s in range(start_ind, end_ind):
                n_rows += len(nbr_clstrs_of[s])
            pq_arr = np.zeros((2*n_rows, 4))
            ctr = 0
            for s in range(start_ind, end_ind):
                c_id = clstrs[s]
                Utilde_s = Utilde[c_id]
                for m in nbr_clstrs_of[s]:
                    if m == c_id:
                        continue
                    Utilde_m = Utilde[m]
                    mc = merging_cost(c_id, m, Utilde_s, Utilde_m, 
                                      d_e, local_param, intermed_opts)
                    pq_arr[2*ctr,:] = [clstr_dist[c_id],mc,c_id,m]
                    mc = merging_cost(m, c_id, Utilde_m, Utilde_s, 
                                      d_e, local_param, intermed_opts)
                    pq_arr[2*ctr+1,:] = [clstr_dist[m],mc,m,c_id]
                    ctr += 1
            q_.put(pq_arr[:2*ctr,:])
        ###########################################
        
        clstrs = list(range(n))
        nbr_clstrs_of = Utilde
        clstr_dist = local_param.zeta
        q_ = mp.Queue()
        for eta in range(n_times):
            self.log('Iter#%d: Match and merge' % eta)
            proc = []
            chunk_sz = int(len(clstrs)/n_proc)
            for p_num in range(n_proc):
                proc.append(mp.Process(target=target_proc,
                                       args=(p_num,chunk_sz,
                                             q_, clstrs,
                                             nbr_clstrs_of,
                                             clstr_dist),
                                       daemon=True))
                proc[-1].start()

            n_done = 0
            pq = []
            for p_num in range(n_proc):
                pq.append(q_.get())
            
            for p_num in range(n_proc):
                proc[p_num].join()
                
            pq = np.concatenate(pq, axis=0)
            #pq = pq[np.lexsort((pq[:,3],pq[:,2],pq[:,1],pq[:,0]))] # for reproducibility
            pq = pq[np.lexsort((pq[:,3],pq[:,2],pq[:,1]))] # for reproducibility
            
            is_merged = np.zeros(n, dtype=bool)
            max_dist = -1
            for i in range(pq.shape[0]):
                s = int(pq[i,2])
                m = int(pq[i,3])
                if is_merged[s] or is_merged[m]:
                    continue
                max_dist = max(max_dist, pq[i,1])
                n_C_s = n_C[s]
                n_C_m = n_C[m]
                
                n_C[m] += n_C_s
                c[c==s] = m
                Utilde[m] = Utilde[m].union(Utilde[s])
                clstr_dist[m] = pq[i,1]
                n_C[s] = 0
                Utilde[s] = None
                clstr_dist[s] = 0
                
                is_merged[s] = True
                is_merged[m] = True
            
            # Update list of clstrs and their nbring clstrs
            clstrs = np.where(n_C>0)[0].tolist()
            nbr_clstrs_of = []
            for m in clstrs:
                nbr_clstrs = c[list(Utilde[m])] # may contain duplicates
                nbr_clstrs_of.append(set(nbr_clstrs))
            self.log('Done. Max cluster distortion is %0.3f' % np.max(clstr_dist), log_time=True)
            
        q_.close()
        return c, n_C
    
    def fit(self, d, U, local_param, intermed_opts):


        n = U.shape[0]
        if (intermed_opts['eta_min'] > 1) or (intermed_opts['c'] is not None):
            if intermed_opts['c'] is None:
                neigh_ind = U
                if intermed_opts['algo'] == 'best':
                    c, n_C = self.best(d, U, local_param, intermed_opts)
                else:
                    c, n_C = self.match_n_merge(d, U, local_param, intermed_opts)
            else:
                c = intermed_opts['c']
                n_C = np.zeros(c.shape[0], dtype=int)
                for i in range(c.shape[0]):
                    n_C[i] = np.sum(c==i)

            self.log('Pruning and cleaning up.')
            # Prune empty clusters
            non_empty_C = n_C > 0
            M = np.sum(non_empty_C)
            old_to_new_map = np.arange(n)
            old_to_new_map[non_empty_C] = np.arange(M)
            c = old_to_new_map[c]
            n_C = n_C[non_empty_C]

            # Construct a boolean ar ray C s.t. C[m,i] = 1 if c_i == m, 0 otherwise
            C = csr_matrix((np.ones(n), (c, np.arange(n))),
                           shape=(M,n), dtype=bool)

            # Compute intermediate views
            intermed_param = copy.deepcopy(local_param)
            if intermed_param.Psi_i is not None:
                intermed_param.Psi_i = local_param.Psi_i[non_empty_C,:]
            if intermed_param.Psi_gamma is not None:
                intermed_param.Psi_gamma = local_param.Psi_gamma[non_empty_C,:]
            if intermed_param.b is not None:
                intermed_param.b = intermed_param.b[non_empty_C]
            if intermed_param.Psi is not None:
                intermed_param.Psi = local_param.Psi[non_empty_C,:]
            if intermed_param.mu is not None:
                intermed_param.mu = intermed_param.mu[non_empty_C,:]
            if intermed_param.model is not None:
                intermed_param.model = local_param.model[non_empty_C]

            # Compute Utilde_m
            Utilde = C.dot(U)
            np.random.seed(42)
            intermed_param.noise_seed = np.random.randint(M*M, size=M)
            self.log('Done.', log_time=True)
        else:
            c = np.arange(n, dtype=int)
            C = csr_matrix((np.ones(n), (c, np.arange(n))),
                           shape=(n,n), dtype=bool)
            n_C = np.ones(n, dtype=int)
            Utilde = U.copy()
            intermed_param = copy.deepcopy(local_param)
            np.random.seed(42)
            intermed_param.noise_seed = np.random.randint(n*n, size=n)
            
        self.C = C
        self.c = c
        self.n_C = n_C
        self.Utilde = Utilde
        self.intermed_param = intermed_param