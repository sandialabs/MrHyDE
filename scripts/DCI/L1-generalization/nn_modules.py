import torch
import torch.nn as nn
import torch.nn.functional as F

# code copied from https://github.com/karpathy/pytorch-made/blob/master/made.py
# first step is to create a masked linear module

class MaskedLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True,mask=None):
        super().__init__(in_features,out_features,bias)

        # register buffer creates a parameter value that is saved
        # but not updated or trained by the optimizer
        self.register_buffer('mask',torch.ones(out_features,in_features))
        # the mask is a matrix of 1's and 0's same size as the weight matrix
        
        if mask!=None:
            self.set_mask(mask)
        
    def set_mask(self,mask):
        # 'self.mask' is the buffer variable we registered
        self.mask.data.copy_(mask)
    
    def forward(self,this_input):
        return F.linear(this_input,self.mask * self.weight, self.bias)


# method generates autoregressive masks
def generate_mask_map(nmasks, d_in, nh_list,random_mask=False,permutation=None):
    mask_maps = []
    for m in range(nmasks):
        this_map = {}
        # assign initial permutation
        if permutation==None:
            this_map[-1] = torch.arange(d_in)
        elif permutation=='random':
            this_map[-1] = torch.arange(d_in)[np.random.choice(np.arange(d_in),size=d_in,replace=False)]
        elif type(permutation)==list:
            this_map[-1] = permutation[m]
        else:
            raise TypeError('Type of permutation not accepted.')
            
        # assign random indices to each neuron in hidden layer
        if random_mask:
            for l,nh in enumerate(nh_list):
                this_map[l] = torch.randint(this_map[l-1].min().item(),d_in-1,
                                            size=[nh,])
        else:
            for l,nh in enumerate(nh_list):
                this_max = max(1,d_in-1)
                this_min = min(1,this_map[l-1].min().item())
                this_map[l] = torch.arange(nh) % this_max +this_min

        # save the mask map
        mask_maps.append(this_map)
        
    return mask_maps



class LocScale_MaskedFF(nn.Module):
    '''
    Explanation of class
    mask_map (dict): dictionary which contains the a tensor of autoregressive indexes assigned
    assigned to each hidden unit of a given layer. Keys should contain:
    {-1: first and last index assignment (tensor), (i): indices of hidden units in layer i (tensor))}
    inverse (boolean): if True then z->x, else x->z
    '''
    def __init__(self,mask_map,inverse=False):
        
        # initializes the super class "torch.nn.Module"
        # which does not initialize itself
        super().__init__()
        
        # If inverse = true then infer z->x
        self.inverse = inverse
        
        # saves the initial parameters
        self.mask_map = mask_map
        self.d_in = len(self.mask_map[-1])
        self.d_out = len(self.mask_map[-1])
        self.nL = len(self.mask_map)-1
        
        # generate a list of mask matrices from mask map
        mask_matrix_list = self.generate_mask_matrices()
        
        # generate a location NN and a scale NN
        for net in ['loc_net','scale_net']:
            # create the NN architecture
            blocks = []

            # first block and all inner layers w/ ReLU
            for l in range(self.nL):
                prev_nH = len(self.mask_map[l-1])
                this_nH = len(self.mask_map[l])
                this_mask_mat = mask_matrix_list[l]
                blocks += [MaskedLinear(prev_nH,this_nH,mask=this_mask_mat),
                           nn.ReLU()]

            # outer layer (doesn't include ReLU)
            prev_nH = len(self.mask_map[l])
            this_mask_mat = mask_matrix_list[-1]
            blocks += [MaskedLinear(prev_nH,self.d_out,mask=this_mask_mat)]

            # define the full sequential net
            if net == 'loc_net':
                self.loc_net = nn.Sequential(*blocks)
            elif net == 'scale_net':
                self.scale_net = nn.Sequential(*blocks)                
        
    def forward(self,y):
        # if inverse, then forward map goes z->x
        if self.inverse:
            output = torch.exp(self.scale_net(y))*y+self.loc_net(y)        
        else:
            # generally, we are looking for g:x->z, because we know prob_z
            output = torch.exp(-self.scale_net(y))*(y-self.loc_net(y))
        return output
    
    
    def backward(self,u):
        y = torch.zeros_like(u)
        mask_conditional_order = self.mask_map[-1] # get conditional order
        for i_col in range(u.shape[1]):
            this_ind_mask = torch.zeros_like(y)
            this_ind_mask[:,mask_conditional_order==i_col] = 1.
            mus = self.loc_net(y)
            betas = self.scale_net(y)

            if self.inverse:
                y = y + this_ind_mask*(torch.exp(-betas)*(u-mus))
            else:
                y = y + this_ind_mask*(torch.exp(betas)*u+mus)


        
        # compute log det
        if self.inverse:
            logdet = betas.sum(-1)
        else:
            logdet = -betas.sum(-1)
            
        return y, logdet


    
    def generate_mask_matrices(self):
        '''Computes the mask matrix from the mask map using identifying number'''
        this_map = self.mask_map
        
        this_mask = [ this_map[l-1][None,:] <= this_map[l][:,None] for l in range(self.nL) ]
        this_mask.append(this_map[self.nL-1][None,:] < this_map[-1][:,None])
        return this_mask

    
class AutoFlow(nn.Module):
    
    def __init__(self,d_in,flow_layers,nn_depth_design,inverse=False):
        
        # initializes the super class "torch.nn.Module"
        # which does not initialize itself
        super().__init__()
        
        # are we learning f:z->x or g:x->z?
        self.inverse = inverse
        
        # saves the initial parameters
        self.d_in = d_in
        self.flow_nL = flow_layers
        self.nn_depth = nn_depth_design
        
        # generate mask map for each layer
        # permute the mask each time 
        # so that the conditional structure is reversed
        mask_map_permutes = []
        for layer in range(self.flow_nL):
            this_val = torch.arange(self.d_in) 
            if layer%2==1:
                this_val = torch.flip(this_val,(0,))
            mask_map_permutes.append(this_val)
            
        self.mask_maps = generate_mask_map(self.flow_nL,self.d_in,self.nn_depth,
                                           permutation=mask_map_permutes)
                
        # generate a series of LocScale and BatchNorm layers
        blocks = []
        norm_blocks = []
        for layer in range(self.flow_nL):
            # create the NN architecture for each layer using LocScaleFF
            blocks.append(LocScale_MaskedFF(self.mask_maps[layer], inverse=self.inverse))
            norm_blocks.append(nn.BatchNorm1d(self.d_in))
        
        self.flow = nn.Sequential(*blocks)
#         self.batch_norms = nn.Sequential(*norm_blocks)
        
    def forward(self,u):
        output = u
        for l in range(self.flow_nL):
#             output = self.batch_norms[l](output)
            output = self.flow[l](output)
        return output
    
    def sequential_backward(self,u,known_y=None,all_steps=False):
        ys = [u]
        logdet = torch.zeros(u.shape[0])
        for l in range(self.flow_nL):
            this_y = ys[l]
            new_y, this_det = self.flow[-l-1].backward(this_y)
            
            # add new y-values to list
            ys.append(new_y)
            
            # accumulate logdet: (+/-) built into backward
            logdet += this_det
            
        # return all steps or just final value
        output = ys[-1] if not all_steps else ys
                
        return output,logdet


    
    def sequential_forward(self,u,all_steps=False):
        ys = [u]
        logdet = torch.zeros(u.shape[0])
        for l,flow_layer in enumerate(self.flow):
            this_y = ys[l]            
#             # batch norm layer
#             this_y = self.batch_norms[l](ys[l])
#             batch_weights = get_batch_norm_scale(self.batch_norms[l])
#             logdet += torch.log(torch.abs(batch_weights).prod())
            
            # flow layer
            these_betas = flow_layer.scale_net(this_y)

            # if inverse, i.e., f:z->x, opposite of g:x->z.
            if self.inverse:
                these_betas *= -1.            
            
            # add new y-values to list
            ys.append(flow_layer(this_y))
            
            # accumulate logdet: for g:x->z -(ba+bb)
            logdet -= these_betas.sum(-1)
            
        # return all steps or just final value
        output = ys[-1] if not all_steps else ys


                
        return output,logdet

    
class ReversibleBatchNorm1d(nn.BatchNorm1d):
    '''
    ReversibleBatchNorm1d modifies BatchNorm1d with a inverse method so that the
    layer can be reversed in invertible neural networks. In addition, it adds a scale method
    that can be used to acquire the scale term for computing the log-determinant contribution
    to transformations of probability.
    
    Note on tracking running stats: it is important to note that running stats are only tracked
    in forward passes, not in inverse passes. Running stats must be tracked in order
    for inverse passes to be computed correctly.
    '''
    def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,
                 device=None,dtype=None):
        super().__init__(num_features,eps=eps,momentum=momentum,affine=affine,
                         track_running_stats=True,device=device,dtype=dtype)
        
        # define training batch mean and var
        ## Note: these might need to have device attached to them for GPU?
        self._train_bmean = None
        self._train_bvar = None
    
    def forward(self, input):
        # get axes to reduce
        axes_red = self._check_input_dim(input)
        
        # get forward from BatchNorm1d
        output = super().forward(input)
        
        # if in training mode, save batch mean and var
        if self.training:    
            self._train_bmean = input.mean(axis=axes_red)
            self._train_bvar = input.var(axis=axes_red,unbiased=False)
            
        return output
    
    def inverse(self,input):
        # get axes to expand
        axes_exp = self._check_input_dim(input)
        
        # get mean and variance
        if not self.training:
            # use running mean and var when in eval mode
            mean = self.running_mean
            var = self.running_var            
        else:
            # when training, use batch mean and var from forward pass
            mean = self._train_bmean
            var = self._train_bvar
        
        # get weights, bias and epsilon
        w,b = (self.weight,self.bias) if self.affine else torch.tensor((1.,0.))
        eps = self.eps        

        # broadcast parameters to correct dimension        
        for ax in axes_exp:
            mean,var = (mean.unsqueeze(ax),var.unsqueeze(ax))
            w,b = (w.unsqueeze(ax),b.unsqueeze(ax))
        
        # compute inverse
        output = (input - b)/w * torch.sqrt(var+eps) + mean
        return output
    
    def scale(self,input,inverse=False):
        axes_red = self._check_input_dim(input)
        if self.training:
            # scale is determined by forward mapping
            if not inverse:
                var = input.var(axis=axes_red,unbiased=False)
            else:
                # if inverse, used the saved training batch var
                var = self._train_bvar
        else:
            var = self.running_var
            
        # compute determinant scale parameter
        out_scale = self.weight/torch.sqrt(var+self.eps)
        if inverse:
            out_scale = 1/out_scale
        return out_scale
    
    def _check_input_dim(self,input):
        if input.dim()==2:
            axes_red = [0]
        elif input.dim()==3:
            axes_red = [0,2]
        else:
            raise ValueError(
                'Wrong shape: Expected 2D or 3D input (got {}D).'.format(input.dim())
            )
        return axes_red
        
class ReversibleBatchMinMaxScaler1d(nn.BatchNorm1d):
    '''
    ReversibleBatchMinMaxScaler1d loosely normalizes the data to [0,1].
    Since data is assumed to be random, it is not guaranteed that the sample min / max of the data
    is the min / max of the distribution. Thus, learned parameters are introduced to allow flexibility
    in the scaling. This leads to a normalization that is not guaranteed to be strictly within the interval
    [0,1], but [b,m+b], where [m,b] are the learned parameters. These are initialized as [1,0], respectively.
    
    '''
    def __init__(self,num_features,eps=1e-9,affine=True,momentum=0.1,
                 device=None,dtype=None):
        super().__init__(num_features,eps=eps,momentum=momentum,affine=affine,
                         track_running_stats=False,device=device,dtype=dtype)
        
        # define training batch min and max
        self._train_bmin = None
        self._train_bmax = None
        
        # register buffers for running min and max
        self.register_buffer('running_min',None)
        self.register_buffer('running_max',None)
        self.num_batches_tracked = torch.tensor(0,dtype=torch.long)
    
    def forward(self, input):
        # get axes to reduce
        axes_red = self._check_input_dim(input)
        
        # calculate running estimates
        if self.training:
            # get this batch min and max
            bmin = input
            bmax = input
            for ax in axes_red:
                bmin = bmin.min(axis=ax,keepdim=True)[0]
                bmax = bmax.max(axis=ax,keepdim=True)[0]
            bmin,bmax = bmin.squeeze(),bmax.squeeze()
            self._train_bmin = bmin
            self._train_bmax = bmax
            
            # save the running stats
            if self.num_batches_tracked == 0:
                self.running_min = bmin.detach()
                self.running_max = bmax.detach()
            else:
                if self.momentum is None: # use cumulative min or max
                    self.running_min = torch.minimum(self.running_min,bmin.detach())
                    self.running_max = torch.maximum(self.running_max,bmax.detach())                
                else:
                    # update min and max
                    self.running_min = self.momentum*bmin.detach()+(1-self.momentum)*self.running_min
                    self.running_max = self.momentum*bmax.detach()+(1-self.momentum)*self.running_max                    
            self.num_batches_tracked += 1
            
        else:
            bmin = self.running_min
            bmax = self.running_max
        
        # get forward by applying normalization with batch stats
        # expand to correct dimensions
        w,b = (self.weight,self.bias) if self.affine else torch.tensor((1.,0.))
        eps = self.eps
        for ax in axes_red:
            bmin,bmax = (bmin.unsqueeze(ax),bmax.unsqueeze(ax))
            w,b = (w.unsqueeze(ax),b.unsqueeze(ax))
        
        # compute normalization
        output = w*(input - bmin)/(bmax - bmin + eps) + b
        return output
    
    def inverse(self,input):
        # get axes to expand
        axes_exp = self._check_input_dim(input)
        
        # get mean and variance
        if not self.training:
            # use running mean and var when in eval mode
            bmin = self.running_min
            bmax = self.running_max            
        else:
            # when training, use batch mean and var from forward pass
            bmin = self._train_bmin
            bmax = self._train_bmax
        
        # get weights, bias and epsilon
        w,b = (self.weight,self.bias) if self.affine else torch.tensor((1.,0.))
        eps = self.eps        

        # broadcast parameters to correct dimension        
        for ax in axes_exp:
            bmin,bmax = (bmin.unsqueeze(ax),bmax.unsqueeze(ax))
            w,b = (w.unsqueeze(ax),b.unsqueeze(ax))
        
        # compute inverse
        output = (input - b)/w * (bmax - bmin + eps) + bmin
        return output
    
    def scale(self,input,inverse=False):
        axes_red = self._check_input_dim(input)
        if self.training:
            # scale is determined by forward mapping
            if not inverse:
                for ax in axes_red:
                    bmin = input.min(axis=ax,keepdim=True)[0]
                    bmax = input.max(axis=ax,keepdim=True)[0]
                bmin,bmax = bmin.squeeze(),bmax.squeeze()
            else:
                # if inverse, used the saved training batch min/max
                bmin = self._train_bmin
                bmax = self._train_bmax
        else:
            bmin = self.running_min
            bmax = self.running_max
            
        # compute determinant scale parameter
        out_scale = self.weight/(bmax-bmin+self.eps)
        if inverse:
            out_scale = 1/out_scale
        return out_scale
    
    def _check_input_dim(self,input):
        if input.dim()==2:
            axes_red = [0]
        elif input.dim()==3:
            axes_red = [0,2]
        else:
            raise ValueError(
                'Wrong shape: Expected 2D or 3D input (got {}D).'.format(input.dim())
            )
        return axes_red
    
class AutoFlowBN(nn.Module):
    def __init__(self,dim,nlayers,base_dist,inner_net_design=3*[15]):
        super().__init__()
        self.nL = nlayers
        self.flow_layers = nn.ModuleList([AutoFlow(dim,2,inner_net_design) for i in range(self.nL)])
        self.batch_norms = nn.ModuleList([ReversibleBatchNorm1d(dim) for i in range(self.nL)])
        
        # this base dist
        # assumed to be an iid distribution
        self.base_dist = base_dist 
        
    def forward(self,x):
        z = x
        logdet = torch.zeros(z.shape[0])
        for flow,norm in zip(self.flow_layers,self.batch_norms):
            # get flow transforms
            z,this_logdet = flow.sequential_forward(z)
            logdet += this_logdet
            
            # get the batchnorm transforms
            this_scale = norm.scale(z) # note, make sure to run scale first!
            logdet += torch.log(torch.abs(this_scale.prod()))
            z = norm.forward(z)
        
        return z, logdet
    
    def log_prob(self,x):
        this_z,logdet = self.forward(x)
        out = self.base_dist.log_prob(this_z).sum(axis=-1)
        out += logdet
        return out
    
    
    def inverse(self,z):
        x = z
        logdet = torch.zeros(x.shape[0])
        
        for i in range(self.nL):
            l = -i-1 # reverse order layers
            
            # get the batchnorm transforms
            this_scale = self.batch_norms[l].scale(x,inverse=True)
            logdet += torch.log(torch.abs(this_scale.prod()))
            x = self.batch_norms[l].inverse(x)            
            
            # get flow transforms
            x,this_logdet = self.flow_layers[l].sequential_backward(x)
            logdet += this_logdet
        
        return x,logdet
                            