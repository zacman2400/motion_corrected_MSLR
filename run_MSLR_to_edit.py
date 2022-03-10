import cupy
import sigpy as sp
import torch
from motion_correction_mslr26 import gen_MSLR,gen_template, adj_field_solver,for_field_solver,train_interp_field, MultiScaleLowRankRecona, MultiScaleLowRankRecon

#generate template
def gen(ksp,coord,dcf,T,RO):
spokes_per_bin=ksp.shape[1]//T
im_template,mps0,kspa,coorda,dcfa=gen_template(ksp,coord,dcf,RO,spokes_per_bin)
return im_template,mps0,kspa,coorda,dcfa

#solve for low res adjoint motion fields (template-->moving)
def LRadjoint(T,RO,rank,scale,block_size_adj,iter_adj,ksp,coord,dcf):
    im_testa,mps0,kspa,coorda,dcfa=gen(ksp,coord,dcf,T,RO) 
    block_size_for=block_size_adj
    deformL_param_adj0,deformR_param_adj0,deformL_param_for0,deformR_param_for0,block_torch0,ishape0a=gen_MSLR(T,rank,block_size_adj,block_size_for,scale,mps0)
    all0=np.abs(im_testa).max()
    im_testa=np.reshape(im_testa,[1,mps0.shape[1],mps0.shape[2],mps0.shape[3]])
    deformL_param_adj0,deformR_param_adj0=adj_field_solver(deformL_param_adj0,deformR_param_adj0,deformL_param_for0,deformR_param_for0,(im_testa)/all0,kspa/all0,coorda,dcfa,mps0,iter_adj,RO,block_torch0,ishape0a,0,T,scale,mps0,spokes_per_bin,20,10)
    for i in range(3):
        np.save('deformL_param_adj0IPF'+str(i),deformL_param_adj0[i].detach().cpu().numpy())
        np.save('deformR_param_adj0IPF'+str(i),deformR_param_adj0[i].detach().cpu().numpy())
    return deformL_param_adj0,deformR_param_adj0,mps0,block_torch0,ishape0a
    
#interpolate low res motion fields
def interpolation(T,RO,rank,scale,block_size_adj,iter,block_torch0,ishape0a,deformL_param_adj0,deformR_paramadj0,mps0,ksp,coord,dcf
spokes_per_bin=ksp.shape[1]//T
im_template,mps1,kspa,coorda,dcfa=gen_template(ksp,coord,dcf,RO,spokes_per_bin)  
high_res_inter=scale
deformL_param_adj1,deformR_param_adj1,deformL_param_for1,deformR_param_for1,block_torch1,ishape1a=gen_MSLR(T,rank,block_size_adj,block_size_for,scale,mps1)
#interpolate low resolution deformation fields-->full resolution deformation fields
old_res=mps0
new_res=mps1
import random
deformL_param_adj=[]
deformR_param_adj=[]
for i in range(3):
    deformL_param_adj.append(torch.from_numpy(deformL_param_adj0[i].detach().cpu().numpy()).cuda())
    deformR_param_adj.append(torch.from_numpy(deformR_param_adj0[i].detach().cpu().numpy()).cuda())
 return deformL_param_for1,deformR_param_for1,block_torch1,ishape1a,im_template,kspa,coorda,dcfa,mps1

train_interp_field(im_template,new_res,old_res,deformL_param_adj,deformR_param_adj,deformL_param_adj1,deformR_param_adj1,deformL_param_for0,deformR_param_for0,deformL_param_for1,deformR_param_for1,iter,T,block_torch0,ishape0a,block_torch1,ishape1a,high_res_interp)
for i in range(3):
    np.save('deformL_param_adj1IPF'+str(i),deformL_param_adj1[i].detach().cpu().numpy())
    np.save('deformR_param_adj1IPF'+str(i),deformR_param_adj1[i].detach().cpu().numpy())

#solve for full resolution forward (moving-->ref)
def for(deformL_param_for1,deformR_param_for1,block_torch1,ishape1a,im_template,scale,T)
deformL_param_adja=[]
deformR_param_adja=[]
for i in range(3):
    deformL_param_adja.append(torch.from_numpy(np.load('deformL_param_adj1IPF'+str(i)+'.npy','r+')).cuda())
    deformR_param_adja.append(torch.from_numpy(np.load('deformR_param_adj1IPF'+str(i)+'.npy','r+')).cuda())
all0=np.abs(im_template).max()
iter=50
for_field_solver(deformL_param_adja,deformR_param_adja,deformL_param_for1,deformR_param_for1,im_template/all0,mps1,iter,block_torch1,ishape1a,0,T,scale,mps1,weight_MSE=1e-1)

#run motion corrected MSLR
L,R=MultiScaleLowRankRecon(kspa[:,:,:], coorda[:,:], dcfa[:,:], mps1, T, 1e-8, ishape1a,deformL_param_adj1 , deformR_param_adj1,deformL_param_for1,deformR_param_for1,block_torch1,
             blk_widths=[64,128], alpha=1, beta=.5,sgw=None,
             device=0, comm=None, seed=0,
             max_epoch=20, decay_epoch=20, max_power_iter=5,
             show_pbar=True).run()


    
    

    