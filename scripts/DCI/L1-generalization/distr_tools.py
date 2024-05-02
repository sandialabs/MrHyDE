import numpy as np
import scipy.stats as sps

class mixture_dist:
    '''
    mixture_dist defines a 1D pdf that is a mixture of multiple distributions
    pdfs: a list of 1D scipy.stats-like frozen pdf objects
    weights: a list of weights, one for each pdf
    multi: optional argument tells the mixture class that the pdfs are multi-dimensional
    '''
    def __init__(self,pdfs,weights,multi=False):
        # check that weights are appropriate
        if np.sum(weights)!=1:
            try:
                this_w = np.array(weights,dtype=float)
                this_w[-1] = 1-np.sum(this_w[0:-1])
                self.weights = this_w
            except:
                raise ValueError('Weights must sum to 1')
        else:
            self.weights = weights
            
        self.pdfs = pdfs
        self.multi = multi
    
    def pdf(self,x):
        if self.multi==True:
            out = np.zeros_like(self.pdfs[0].pdf(x))
        else:
            out = np.zeros_like(x)
        for weight,distr in zip(self.weights,self.pdfs):
            out += weight*distr.pdf(x)
        return out
    
    def logpdf(self,x):
        out = self.pdf(x)
        return np.log(out)
    
    def rvs(self,size=1,random_state=None):
        if (random_state == None) or (type(random_state)==int):
            rng = np.random.default_rng(random_state)
        elif type(random_state)==np.random.Generator:
            rng = random_state
        else:
            raise ValueError ('random_state must be an integer or numpy.random.Generator object.')            
            
        m = len(self.weights)
        which_pdf = rng.choice(np.arange(m),size=size,p=self.weights)
        
        if self.multi==True:
            try:
                iter(size)
                this_shape = tuple(size)
            except TypeError:
                this_shape = (size,)
                
            out_sample = np.ones(this_shape+tuple([self.pdfs[0].mean.shape[0]]))*np.NaN
        else:
            out_sample = np.ones(size)*np.NaN
        
        for i,pdf in enumerate(self.pdfs):
            these_sample_idx = which_pdf==i
            gen_size = np.sum(these_sample_idx)
            #print('For PDF ', i,' generate: ',gen_size, ' samples')
            out_sample[these_sample_idx] = pdf.rvs(gen_size,random_state=rng)
        
        return out_sample
    
class mv_iid_dist:
    '''
    Defines an iid distribution with potentially different distributions in each dimension
    '''
    def __init__(self,pdfs):
        self.d = len(pdfs)
        self.pdfs = pdfs
        
    def logpdf(self,x):
        # check appropriate shape
        if len(x.shape)!=2:
            try:
                this_x = x.reshape(-1,self.d)
            except:
                raise ValueError ('Shape must be castable to (n x d)')
        else:
            this_x = x
        
        # compute pdf
        output = np.zeros(this_x.shape[0])
        for i,pdf in enumerate(self.pdfs):
            output += pdf.logpdf(this_x[:,i])
        return output
    
    def pdf(self,x):
        output = np.exp(self.logpdf(x))
        return output
    
    def rvs(self,size=1,random_state=None):
        if (random_state == None) or (type(random_state)==int):
            rng = np.random.default_rng(random_state)
        elif type(random_state)==np.random.Generator:
            rng = random_state
        else:
            raise ValueError ('random_state must be an integer or numpy.random.Generator object.')
        
        output = np.empty([*np.atleast_1d(size),self.d])
        for i,pdf in enumerate(self.pdfs):
            output[...,i] = pdf.rvs(size)
        return output
        
            
    
