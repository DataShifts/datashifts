import torch
import numpy as np
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor
from torch import Tensor
from numpy import ndarray
from typing import Callable, Union, Optional

N_min=3

def default_distance_expansion(A, B, KeOps):
    #Expanding the dimensions of two tensors for calculating the matrix of default distance 
    dim=A.dim()
    if KeOps:
        packer=LazyTensor
    else:
        packer=lambda x:x
    if dim == 1:
        A_expand = packer(A[:, None, None])  # (N,1,1)
        B_expand = packer(B[None, :, None])  # (1,M,1)
    elif dim >= 2:
        A_expand = packer(A.unsqueeze(dim-1))  # (..B,N,1,D)
        B_expand = packer(B.unsqueeze(dim-2))  # (..B,1,M,D)
    return A_expand, B_expand

def Euclidean_distance(A, B, KeOps=True):
    A_expand, B_expand=default_distance_expansion(A, B, KeOps)
    return ((A_expand - B_expand) ** 2).sum(-1) ** (1 / 2)

def Manhattan_distance(A, B, KeOps=True):
    A_expand, B_expand=default_distance_expansion(A, B, KeOps)
    return ((A_expand - B_expand).abs()).sum(-1)

def Chebyshev_distance(A, B, KeOps=True):
    A_expand, B_expand=default_distance_expansion(A, B, KeOps)
    return ((A_expand - B_expand).abs()).max(-1)

def W1_deb(x1, x2, w1, w2, eps=0.01):
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=eps, debias=True, scaling=0.9)
    index1=int(x1.shape[-2]/2)
    index2=int(x2.shape[-2]/2)
    dim_batch=len(w1.shape)-1
    idx11=(slice(None),)*dim_batch+(slice(None,index1),)
    idx12=(slice(None),)*dim_batch+(slice(index1,None),)
    idx21=(slice(None),)*dim_batch+(slice(None,index2),)
    idx22=(slice(None),)*dim_batch+(slice(index2,None),)
    x11,x12=x1[idx11],x1[idx12]
    x21,x22=x2[idx21],x2[idx22]
    w11,w12=w1[idx11],w1[idx12]
    w21,w22=w2[idx21],w2[idx22]
    w11=w11/(w11.sum(axis=-1).unsqueeze(-1))
    w12=w12/(w12.sum(axis=-1).unsqueeze(-1))
    w21=w21/(w21.sum(axis=-1).unsqueeze(-1))
    w22=w22/(w22.sum(axis=-1).unsqueeze(-1))
    W_x12_1=loss(w11, x11, w21, x21).item()
    W_x12_2=loss(w12, x12, w22, x22).item()
    W_x11=loss(w11, x11, w12, x12).item()
    W_x22=loss(w21, x21, w22, x22).item()
    W1_deb=abs(W_x12_1**2/2+W_x12_2**2/2-W_x11**2/2-W_x22**2/2)**(1/2)
    return W1_deb

def ensure_no_grad(x, dataname):
    if isinstance(x, Tensor):
        if x.requires_grad:
            return x.detach(),True 
        else:
            return x,False
    elif isinstance(x, ndarray):
        return x,False
    else:
        raise TypeError("")

def check_class(x1, x2, y1, y2, weights1, weights2):
    requires_grad={}
    x1_,requires_grad["x1"]=ensure_no_grad(x1, "x1")
    x2_,requires_grad["x2"]=ensure_no_grad(x2, "x2")
    y1_,requires_grad["y1"]=ensure_no_grad(y1, "y1")
    y2_,requires_grad["y2"]=ensure_no_grad(y2, "y2")
    if weights1==None:
        weights1_,requires_grad["weights1"]=None,None
    else:
        weights1_,requires_grad["weights1"]=ensure_no_grad(weights1, "weights1")
    if weights2==None:
        weights2_,requires_grad["weights2"]=None,None
    else:
        weights2_,requires_grad["weights2"]=ensure_no_grad(weights2, "weights2")
    return x1_, x2_, y1_, y2_, weights1_, weights2_, requires_grad

def check_coupling_format(data1, data2, Dis, axis, N_max, dataname):
    
    if Dis!="L2":
        raise ValueError("Currently the “Dis” value is only supported for “L2”, other distances will be supported in subsequent releases.")
    
    if Dis in ["L1","L2","inf"]:
        original_dis=True
    elif isinstance(Dis, Callable):
        original_dis=False
    elif isinstance(Dis, str):
        raise ValueError("")
    else:
        raise TypeError("")
    
    shape_data1=tuple(data1.shape)
    shape_data2=tuple(data2.shape)
    shape_batch=None
    N_data1=None
    N_data2=None
    if original_dis:
        if len(shape_data1)>=2 and len(shape_data2)>=2 and shape_data1[:-2]==shape_data2[:-2] and shape_data1[-1]==shape_data2[-1]:
            shape_batch=shape_data1[:-2]
            N_data1=shape_data1[-2]
            N_data2=shape_data2[-2]
            if N_data1<N_min or N_data2<N_min:
                raise ValueError("")
        else:
            raise ValueError("")
        
        dim=shape_data1[-1]
        if len(shape_batch)>0:
            batchs=(torch.cumprod(torch.tensor(shape_batch),dim=0)).item()
        else:
            batchs=1
        estimated_space=4*min(N_data1,N_max)*min(N_data2,N_max)*dim*batchs
        if torch.cuda.is_available():
            allowed_space=1024**3
            if estimated_space<=allowed_space:
                KeOps=False
            else:
                KeOps=True
        else:
            KeOps=True
    else:
        if axis==None:
            raise ValueError("")
        elif not isinstance(axis, int):
            raise TypeError("")
        
        if shape_data1[:axis]==shape_data2[:axis] and shape_data1[axis+1:]==shape_data2[axis+1:]:
            shape_batch=shape_data1[:axis]
            N_data1=shape_data1[axis]
            N_data2=shape_data2[axis]
            if N_data1<N_min or N_data2<N_min:
                raise ValueError("")
        else:
            raise ValueError("")
        
        try:
            idx=(slice(None),)*axis+(slice(0,N_min),)
            C=Dis(data1[idx],data2[idx])
        except Exception as e:
            raise ValueError("") from e
        
        if isinstance(C, Tensor):
            KeOps=False
        elif isinstance(C, LazyTensor):
            KeOps=True
        else:
            raise TypeError("")
        
        shape_C=tuple(C.shape)
        if shape_C[:axis]==shape_batch and shape_C[axis:]==(N_min,N_min) and not KeOps:
            pass
        elif shape_C[:axis]==shape_batch and shape_C[axis:]==(N_min,N_min,1) and KeOps:
            pass
        else:
            raise ValueError("")
    return shape_batch, N_data1, N_data2, KeOps, original_dis

def shuffling_and_sampling(weights, N ,N_max, shape_batch, generator, Cuda, verbose, dataname):
    if isinstance(weights, ndarray):
        weights_=torch.tensor(weights,dtype=torch.float32,device="cpu")
    if N>N_max:
        sampling=True
        if verbose:
            print("Data volume of (x%s,y%s,w%s) is larger than parameter N_max=%d, sampling strategy is used."%(dataname,dataname,dataname,N_max))
    else:
        sampling=False
    if weights==None:
        Index=torch.randperm(N, generator=generator,device="cpu")
        if sampling:
            Index=Index[:N_max]
    else:
        if sampling:
            if len(shape_batch)!=0:
                flat_batch=int(torch.prod(torch.tensor(shape_batch)))
                weights_flat = weights.reshape(flat_batch, C)
                Index=torch.multinomial(weights_flat, N_max, replacement=False, generator=generator)
                Index=Index.reshape(*shape_batch, N_max)
            else:
                Index=torch.multinomial(weights, N_max, replacement=False, generator=generator)
            ReIndex=torch.randperm(Index.shape[-1], generator=generator,device=Index.device)
            Index=Index[(slice(None),)*len(shape_batch)+(ReIndex,)]
        else:
            if (weights.min(axis=-1)[0]>=0).all() and (weights.max(axis=-1)[0]>0).any():
                pass
            else:
                raise ValueError("")
            Index=torch.randperm(N, generator=generator,device="cpu")
    return Index

def one_dimension_indexing(data, Index, d):
    if isinstance(data, Tensor):
        if len(Index.shape)==1:
            idx=(slice(None),)*d+(Index.to(data.device),)
            samples=data[idx]
        else:
            samples=torch.gather(data,d,Index.to(data.device))
    else:
        Index_=Index.cpu().numpy()
        if len(Index.shape)==1:
            idx=(slice(None),)*d+(Index_,)
            samples=data[idx]
        else:
            samples=np.take_along_axis(data,Index_,d)
    return samples

def tensorized(data, CUDA):
    if isinstance(data, ndarray):
        return torch.tensor(data, dtype=torch.float32, device="cuda") if CUDA else torch.tensor(data, dtype=torch.float32, device="cpu")
    else:
        return data.to("cuda") if CUDA else data.to("cpu")

def datashifts(
                x1:Union[Tensor,ndarray], 
                x2:Union[Tensor,ndarray], 
                y1:Union[Tensor,ndarray], 
                y2:Union[Tensor,ndarray], 
                Dis_x:Union[str,Callable]="L2", 
                Dis_y:Union[str,Callable]="L2", 
                axis_x:Optional[int]=None, 
                axis_y:Optional[int]=None, 
                weights1:Union[Tensor,ndarray]=None, 
                weights2:Union[Tensor,ndarray]=None, 
                eps:float=0.01, 
                N_max:int=5000, 
                seed:Optional[int]=None, 
                KeOps:Optional[bool]=None, 
                device:Optional[str]=None, 
                verbose:bool=True,
              ):
    #x1_, x2_, y1_, y2_, weights1_, weights2_, requires_grad=check_class(x1, x2, y1, y2, weights1, weights2)
    x1_, x2_, y1_, y2_, weights1_, weights2_=x1, x2, y1, y2, weights1, weights2
    if isinstance(N_max, float) or isinstance(N_max, int):
        N_max=int(N_max)
        if N_max<N_min:
            raise ValueError("")
    else:
        raise TypeError("")
    shape_batch, N1, N2, KeOps_x, original_dis_x=check_coupling_format(x1_, x2_, Dis_x, axis_x, N_max, "x")
    if Dis_y in ["L1","L2","inf"]:
        shape_y1=tuple(y1_.shape)
        if shape_y1[:-1]==shape_batch:
            idx=(slice(None),)*len(shape_y1)+(None,)
            y1_=y1_[idx]
        shape_y2=tuple(y2_.shape)
        if shape_y2[:-1]==shape_batch:
            idx=(slice(None),)*len(shape_y2)+(None,)
            y2_=y2_[idx]
    shape_batch_y, N1_y, N2_y, KeOps_y, original_dis_y=check_coupling_format(y1_, y2_, Dis_y, axis_y, N_max, "y")
    if shape_batch_y!=shape_batch:
        raise ValueError("")
    if N1_y!=N1:
        raise ValueError("")
    if N2_y!=N2:
        raise ValueError("")
    if device in ["cpu","cuda",None]:
        pass
    else:
        raise ValueError("")
    if torch.cuda.is_available() and device!="cpu":
        Cuda=True
    else:
        Cuda=False
    if weights1_!=None:
        shape_weights1=tuple(weights1_.shape)
        if shape_weights1[:-1]!=shape_batch:
            raise ValueError("")
        if shape_weights1[-1]!=N1:
            raise ValueError("")
    if weights2_!=None:
        shape_weights2=tuple(weights2_.shape)
        if shape_weights2[:-1]!=shape_batch:
            raise ValueError("")
        if shape_weights2[-1]!=N2:
            raise ValueError("")
    if not isinstance(verbose, bool):
        raise TypeError("")
    if isinstance(weights1_, Tensor):
        g_device=weights1_.device
    else:
        g_device="cpu"
    g=torch.Generator(device=g_device)
    g.seed()
    if seed!=None:
        g.manual_seed(seed)
    index1=shuffling_and_sampling(weights1_, N1, N_max, shape_batch, g, Cuda, verbose, "1")
    N1_used=index1.shape[-1]
    index2=shuffling_and_sampling(weights2_, N2, N_max, shape_batch, g, Cuda, verbose, "2")
    N2_used=index2.shape[-1]
    if original_dis_x:
        x1_used=one_dimension_indexing(x1_,index1,len(shape_batch))
        x2_used=one_dimension_indexing(x2_,index2,len(shape_batch))
        if weights1_==None:
            weights1_used=torch.ones(shape_batch+(N1_used,),dtype=torch.float32,device="cpu")
        else:
            weights1_used=one_dimension_indexing(weights1_,index1,len(shape_batch))
        if weights2_==None:
            weights2_used=torch.ones(shape_batch+(N2_used,),dtype=torch.float32,device="cpu")
        else:
            weights2_used=one_dimension_indexing(weights2_,index2,len(shape_batch))
        x1_used=tensorized(x1_used,Cuda)
        x2_used=tensorized(x2_used,Cuda)
        weights1_used=tensorized(weights1_used,Cuda)
        weights2_used=tensorized(weights2_used,Cuda)
    else:
        pass
    weights1_used=weights1_used/(weights1_used.sum(axis=-1).unsqueeze(-1))
    weights2_used=weights2_used/(weights2_used.sum(axis=-1).unsqueeze(-1))
    if original_dis_y:
        y1_used=one_dimension_indexing(y1_,index1,len(shape_batch))
        y2_used=one_dimension_indexing(y2_,index2,len(shape_batch))
        y1_used=tensorized(y1_used,Cuda)
        y2_used=tensorized(y2_used,Cuda)
    else:
        pass
    #eps
    try:
        eps=float(eps)
    except:
        raise TypeError("")
    
    covariance_shift=W1_deb(x1_used, x2_used, weights1_used, weights2_used, eps=eps)
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=(eps)**(1/1), potentials=True, debias=False, scaling=(0.9)**(1/1))
    g1,g2=loss(weights1_used, x1_used, weights2_used, x2_used)
    if len(shape_batch)==0 and g1.shape[0]==1:
        g1=g1.squeeze(0)
        g2=g2.squeeze(0)
    g1_=LazyTensor(g1.unsqueeze(-1).unsqueeze(-1))
    g2_=LazyTensor(g2.unsqueeze(-2).unsqueeze(-1))
    weights1_used_=LazyTensor(weights1_used.unsqueeze(-1).unsqueeze(-1))
    weights2_used_=LazyTensor(weights2_used.unsqueeze(-2).unsqueeze(-1))
    Cx=Euclidean_distance(x1_used,x2_used,KeOps=True)
    Cy=Euclidean_distance(y1_used,y2_used,KeOps=True)
    Pi=weights1_used_*(((g1_-Cx+g2_)/eps).exp())*weights2_used_
    concept_shift=(((Pi*Cy).sum(axis=0).sum(axis=0))[0]).item()
    return covariance_shift, concept_shift

def dA_Distance(x1, x2, clf):
    """
    Compute and return the value D, which is defined as the sum of the accuracy of 
    the Logistic Regression model on both classes minus 1.
    
    Explanation:
      - The input x1 represents the sample data for cluster 1 (label assigned as 0), 
        and x2 represents the sample data for cluster 2 (label assigned as 1).
      - Each sample in cluster 1 has a weight of 1/N1, and in cluster 2 a weight of 1/N2.
      - The function uses isinstance() to check the input data type and converts 
        it to a numpy.ndarray (assigning to a new variable) to avoid modifying the original data.
    
    Parameters:
      x1: Sample data of cluster 1, with shape (N1, n_features)
      x2: Sample data of cluster 2, with shape (N2, n_features)
    
    Returns:
      D = (accuracy on cluster 1 + accuracy on cluster 2) - 1
    """
    # Type checking and data conversion
    if not isinstance(x1, ndarray):
        x1_data = np.array(x1.cpu())
    else:
        x1_data = x1.copy()
        
    if not isinstance(x2, ndarray):
        x2_data = np.array(x2.cpu())
    else:
        x2_data = x2.copy()
    
    # Calculate the number of samples in each cluster
    N1 = x1_data.shape[0]
    N2 = x2_data.shape[0]
    
    # Construct the training data X and corresponding labels y
    X = np.vstack((x1_data, x2_data))
    y = np.concatenate((np.zeros(N1), np.ones(N2)))
    
    # Construct the sample weights
    weights1 = np.full(N1, 1.0 / N1)
    weights2 = np.full(N2, 1.0 / N2)
    sample_weight = np.concatenate((weights1, weights2))
    
    # Train the model
    #clf.fit(X, y, sample_weight=sample_weight)
    #The mlp in sklearn does not support defining sample weights, and for the time being requires an equal number of x1 and x2 samples.
    clf.fit(X, y)
    
    # Perform prediction on the training data
    y_pred = clf.predict(X)
    
    # Compute the accuracy on x1 (label 0) and x2 (label 1) separately
    acc_x1 = np.mean(y_pred[:N1] == 0)
    acc_x2 = np.mean(y_pred[N1:] == 1)
    
    # Compute D = (accuracy on cluster 1 + accuracy on cluster 2) - 1
    D = acc_x1 + acc_x2 - 1
    return D

def old_bound(x1, x2, f1, f2, clf):
    concept_shift = min((f1(x1)-f2(x1)).abs().mean().item(),
                       (f1(x2)-f2(x2)).abs().mean().item())
    covariance_shift = dA_Distance(x1, x2, clf)
    return covariance_shift, concept_shift
