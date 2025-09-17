"""
    Performs batched local Parametrization evaluation, transformation and reconstruction.
"""

class BatchParam:
    def __init__(self,
                 algo = 'LPCA',
                 **kwargs):
        self.algo = algo
        self.T = None
        self.v = None
        self.b = None
        # Following variables are
        # initialized externally
        # i.e. by the caller
        self.zeta = None
        self.noise_seed = None
        self.noise_var = 0
        self.noise = None
        
        # For LDLE and RFFLE
        self.Psi_gamma = None
        self.Psi_i = None
        self.phi = None
        self.gamma = None
        self.w = None
        
        # For LPCA and its variants
        self.Psi = None
        self.mu = None
        self.X = None
        self.y = None
        
        # For KPCA, ISOMAP etc
        self.model = None
        self.X = None
        self.y = None
        
        self.add_dim = False
        
    def eval_(self, opts):
        
        if self.algo == 'LDLE' or self.algo == 'RFFLE':
            raise Exception("Not implemented")
            # if self.w is None:
            #     temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
            # else:
            #     w_ = self.w[k,:]
            #     temp = (self.gamma[k,:][np.newaxis,:]*self.phi[mask,:]).dot(w_)
            # n = self.phi.shape[0]
        elif self.algo == 'LPCA':
            temp = torch.dot(self.X[mask,:]-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
            n = self.X.shape[0]
        else:
            temp = self.model[k].transform(self.X[mask,:])
        
        if self.noise_var:
            np.random.seed(self.noise_seed[k])
            temp2 = np.random.normal(0, self.noise_var, (n, temp.shape[1]))
            temp = temp + temp2[mask,:]

        if self.noise is not None:
            temp = temp + self.noise[k, mask, :]
            
        if self.add_dim:
            temp = np.concatenate([temp,np.zeros((temp.shape[0],1))], axis=1)
        
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k,:,:])
            if self.v is not None:
                temp = temp + self.v[[k],:]
            return temp
        
    def reconstruct_(self, opts):
        k = opts['view_index']
        y_ = opts['embeddings']
        if self.algo == 'LDLE' or self.algo=='RFFLE':
            pass
        elif self.algo == 'LPCA':
            temp = np.dot(np.dot(y_-self.v[[k],:], self.T[k,:,:].T),self.Psi[k,:,:].T)+self.mu[k,:][np.newaxis,:]
        else:
            pass
        return temp
    
    def out_of_sample_eval_(self, opts):
        k = opts['view_index']
        X_ = opts['out_of_samples']
        
        if self.algo == 'LDLE' or self.algo=='RFFLE':
            temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
            n = self.phi.shape[0]
        elif self.algo == 'LPCA':
            temp = np.dot(X_-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
            n = self.X.shape[0]
        else:
            temp = self.isomap[k].transform(X_)
        
        if self.noise_var:
            np.random.seed(self.noise_seed[k])
            temp2 = np.random.normal(0, self.noise_var, (n, temp.shape[1]))
            temp = temp + temp2[mask,:]
            
        if self.add_dim:
            temp = np.concatenate([temp,np.zeros((temp.shape[0],1))], axis=1)
        
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k,:,:])
            if self.v is not None:
                temp = temp + self.v[[k],:]
            return temp
    
    def alignment_wts(self, opts):
        beta = opts['beta']
        if beta is None:
            return None
        k = opts['view_index']
        mask = opts['data_mask']
        mu = np.mean(self.X[mask,:], axis=0)
        temp = self.X[mask,:] - mu[None,:]
        w = -np.linalg.norm(temp, 1, axis=1)/beta
        return w
        #p = np.exp(w - np.max(w))
        #p *= (temp.shape[0]/np.sum(p))
        #return p
    def repulsion_wts(self, opts):
        beta = opts['beta']
        if beta is None:
            return None
        k = opts['pt_index']
        far_off_pts = opts['repelling_pts_indices']
        if self.y is not None:
            temp = self.y[far_off_pts,:] - self.y[k,:][None,:]
            w = np.linalg.norm(temp, 2, axis=1)**2
            #temp0 = self.X[far_off_pts,:] - self.X[k,:][None,:]
            #w0 = np.linalg.norm(temp0, 2, axis=1)**2
            #p = 1.0*((w-w0)<0)
            p = 1/(w + 1e-12)
        else:
            p = np.ones(len(far_off_pts))
        return p
