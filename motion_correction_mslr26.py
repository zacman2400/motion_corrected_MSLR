import cupy
import sigpy
import torch
import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from interpol import grid_pull
import random
#from multi_scale_low_rank_image import MultiScaleLowRankImage
print('a')
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class MultiScaleLowRankRecona:
    r"""Multi-scale low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.

    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda,
                 blk_widths=[32, 64, 128], alpha=1, beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=200, decay_epoch=20, max_power_iter=5,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

        self._normalize()

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue.
            coord_t = sp.to_device(self.coord[:self.tr_per_frame], self.device)
            dcf_t = sp.to_device(self.dcf[:self.tr_per_frame], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=self.max_power_iter,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            self.dcf /= max_eig

            # Estimate scaling.
            img_adj = 0
            dcf = sp.to_device(self.dcf, self.device)
            coord = sp.to_device(self.coord, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=range(-self.D, 0), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            self.sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                self.sigma.append(L_j_norm)

        for j in range(self.J):
            self.L[j] *= self.sigma[j]**0.5
            self.R[j] *= self.sigma[j]**0.5

    def _AHyH_L(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

    def _AHy_R(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])

    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return self.L,self.R,self.B
           

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    loss += self._update(t)
                    pbar.set_postfix(loss=loss * self.T / (i + 1))
                    pbar.update()

    def _update(self, t):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # L gradient.
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Precondition.
            g_L_j /= self.J * self.sigma[j] + lamda_j
            g_R_jt /= self.J * self.sigma[j] + lamda_j

            # Add.
            self.L[j] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t


def normalize(mps,coord,dcf,ksp,tr_per_frame):
    mps=mps
   
   # import cupy
    import sigpy as sp
    device=0
    # Estimate maximum eigenvalue.
    coord_t = sp.to_device(coord[:tr_per_frame], device)
    dcf_t = sp.to_device(dcf[:tr_per_frame], device)
    F = sp.linop.NUFFT([mps.shape[1],mps.shape[2],mps.shape[3]], coord_t)
    W = sp.linop.Multiply(F.oshape, dcf_t)

    max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500, device=0,
                            dtype=ksp.dtype,show_pbar=True).run()
    dcf1=dcf/max_eig
    return dcf1

def kspace_scaling(mps,dcf,coord,ksp):
    # Estimate scaling.
    img_adj = 0
    device=0
    dcf = sp.to_device(dcf, device)
    coord = sp.to_device(coord, device)
   
    for c in range(mps.shape[0]):
        print(c)
        mps_c = sp.to_device(mps[c], device)
        ksp_c = sp.to_device(ksp[c], device)
        img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, [mps.shape[1],mps.shape[2],mps.shape[3]])
        img_adj_c *= cupy.conj(mps_c)
        img_adj += img_adj_c


    img_adj_norm = cupy.linalg.norm(img_adj).item()
    print(img_adj_norm)
    ksp1=ksp/img_adj_norm
    return ksp1

#RO=150



#ksp=kspa[:,:,:RO]
##coord=coorda[:,:RO]
#d#cf=dcfa[:,:RO]
def gen_template(ksp,coord,dcf,RO,spokes_per_bin):
    import sigpy as sp
    shape=sp.estimate_shape(coord)
    matrix_dim=np.ones([1,shape[0],shape[1],shape[2]])

    kspa=ksp[:,:,:RO]
    coorda=coord[:,:RO]
    dcfa=dcf[:,:RO]

    #generate sense maps
    import sigpy.mri as mr
    import sigpy as sp
    device=0
    mps = mr.app.JsenseRecon(kspa[:,:], coord=coorda[:], weights=dcfa[:], device=0).run()
   # mps = mr.app.JsenseRecon(kspa, coord=coorda, weights=dcfa, device=0).run()

    print(mps.shape)

    #normalize data

    device=0

    dcfa=normalize(mps,coorda,dcfa,kspa,spokes_per_bin)
    import cupy
    kspa=kspace_scaling(mps,dcfa,coorda,kspa)
    import sigpy
    #P=sigpy.mri.kspace_precond(cupy.array(mps), weights=cupy.array(dcf), coord=cupy.array(coord), lamda=0, device=0, oversamp=1.25)


    T =1
    device=0
    lamda = 1e-8
    blk_widths = [8,16,32]  # For low resolution.
    al=ksp.shape[1]//2
    L_blocks,R_blocks,B = MultiScaleLowRankRecona(kspa[:,:,:], coorda[:,:], dcfa[:,:], mps, T, lamda, device=device, blk_widths=blk_widths).run()

    mpsa=mps

    im_test=np.zeros([1,mpsa.shape[1],mpsa.shape[2],mpsa.shape[3]],dtype=np.complex64)
    temp=0
    for i in range(1):
        for j in range(3):
            temp=temp+B[j](L_blocks[j]*R_blocks[j][i])
        im_test[i]=temp.get()

    im_testa=im_test
    return im_testa,mps,kspa,coorda,dcfa

def inverse_field(u,mps):
    v=torch.zeros_like(u).cuda()
    for i in range(10):
        #print(i)
        torch.cuda.empty_cache()
        v=torch.utils.checkpoint.checkpoint(warp0,-u, v,mps)
    return v

def warp0(img,flow,mps,complex=False):
    img=img.cuda()
    img=torch.reshape(img,[1,3,mps.shape[1], mps.shape[2], mps.shape[3]])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])

    spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
    size=shape
    vectors=[]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors)
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    new_locs=grid.cuda()+flow
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
  #  for i in range(len(shape)):
  #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
   # new_locs = new_locs[..., [2,1,0]]
    new_locsa = new_locs[..., [0,1,2]]
    if complex==True:
        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        im_out=torch.complex(ima_real,ima_imag)
    else:
        im_out=grid_pull(torch.squeeze((img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
     #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
     #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
    #im_out=torch.complex(ima_real,ima_imag)
    return im_out

def warp1(img,flow,mps,complex=False):
    img=img.cuda()
    #img=torch.reshape(img,[1,1,304, 176, 368])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])

    spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
    size=shape
    vectors=[]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors)
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    new_locs=grid.cuda()+flow
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
  #  for i in range(len(shape)):
  #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
   # new_locs = new_locs[..., [2,1,0]]
    new_locsa = new_locs[..., [0,1,2]]
    if complex==True:
        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        im_out=torch.complex(ima_real,ima_imag)
    else:
        im_out=grid_pull(torch.squeeze((img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
     #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
     #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
    #im_out=torch.complex(ima_real,ima_imag)
    return im_out
   # im_out=torch.squeeze(im_out)

    #contrast MSLR, use
def adj_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_template,ksp,coord,dcf,mps,iter_adj,RO,block_torch,ishape,T1,T2,interp,res,spokes_per_bin,weight_dc,weight_smoother): #,conL,conR,block_torchcon,ishapecon):
   # from utils_reg1 import flows,_updateb,f
    from torch.utils import checkpoint
    import torch
    import numpy as np
    import random
    scaler = torch.cuda.amp.GradScaler()
    #readout images and deformation fields during training
    deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    im_tester=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    P=torch.ones([40,1])
    mps=torch.from_numpy(mps).cpu()
   # deform=[]
    deform=[deformL_param_adj[0],deformR_param_adj[0],deformL_param_adj[1],deformR_param_adj[1],deformL_param_adj[2],deformR_param_adj[2]] #,conL[0],conR[0],conL[1],conR[1]] #,deformR_param_adj[1],deformR_param_adj[2]]
    import torch_optimizer as optim
   # optimizer0=torch.optim.Adam([deform[i] for i in range(6)],lr=.01) # , max_iter=1, max_eval=None, tolerance_grad=1e-200, tolerance_change=1e-400, history_size=100, line_search_fn='backtracking') #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)
   # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer0, gamma=0.9)
   # deform[0].requires_grad=True
   # deform[1].requires_grad=True
   # deform[2].requires_grad=True
   # deform[3].requires_grad=True
   # deform[4].requires_grad=True
   # deform[5].requires_grad=True
   # optimizer0=torch.optim.Adam([deform[i] for i in range(6)], lr=.01) #, max_eval=1, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe') 
   # optimizer0=torch.optim.Adam([deform[i] for i in range(4)],.01) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)
    optimizer0=torch.optim.Adam([deform[i] for i in range(6)],.007)
 #  optimizer1=torch.optim.Adam([deform[i] for i in range(12,16)],.01)
  #  optimizer0=torch.optim.Adam([deform[i] for i in range(6,10)],lr=.001) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe') 
    def closure_adj():
        K=random.sample(range(T1,T2), T2-T1)
        print(K)
        for j in range(jo,jo+1):
            #j=91
          #  print(count)
           # print(j)

          #  count0=count%1
          #  L[count0]=np.int(j)
           # print(count)
           # print(j)

          #  print(j)
           # optimizer0.zero_grad()


            deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
            con=cons(conL,conR,j-T1,block_torchcon,ishapecon)
          #  con_real=torch.squeeze(torch.nn.functional.interpolate(torch.real(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
          #  con_imag=torch.squeeze(torch.nn.functional.interpolate(torch.imag(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
           # con=torch.complex(con_real,con_imag)
            all1=torch.abs(con).max()
          
            flowa=deforma.cuda()
            flowa=torch.nn.functional.interpolate(flowa, size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
            #flowa[:,0]=flowa[:,0]*inter
            #flowa[:,1]=flowa[:,1]*3
            #flowa[:,2]=flowa[:,2]*3
           # flowa_rev=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)

           # loss_for=0


            im_test=torch.from_numpy(im_template).cuda().unsqueeze(0)
            con0=con+im_test
            all0=torch.abs(im_test).max()

           # im_test_con=torch.squeeze(im_test) #/all0

            spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
            shape=(mps.shape[1],mps.shape[2],mps.shape[3])
            size=shape
            vectors=[]
            vectors = [ torch.arange(0, s) for s in size ] 
            grids = torch.meshgrid(vectors) 
            grid  = torch.stack(grids) # y, x, z
            grid  = torch.unsqueeze(grid, 0)  #add batch
            grid = grid.type(torch.FloatTensor)
            new_locs=grid.cuda()+flowa.cuda()
            shape=(mps.shape[1],mps.shape[2],mps.shape[3])
          #  for i in range(len(shape)):
          #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
           # new_locs = new_locs[..., [2,1,0]]
            new_locsa = new_locs[..., [0,1,2]]

            ima_real=grid_pull(torch.squeeze(torch.real(con0)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
            ima_imag=grid_pull(torch.squeeze(torch.imag(con0)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)

           # cona_real=grid_pull(torch.squeeze(torch.real(con)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
           # cona_imag=grid_pull(torch.squeeze(torch.imag(con)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
            im_no_con=grid_pull(torch.squeeze(torch.abs(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
           # ima_imag=grid_pull(torch.squeeze(torch.imag(im_test_con)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
           # ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='zeros', align_corners=True)
           # ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='zeros', align_corners=True)
            im_out=torch.complex(ima_real,ima_imag) #+torch.complex(cona_real,cona_imag)


           # loss_rev=torch.norm(diff_rev,2)

            tr_per_frame=spokes_per_bin
            tr_start=tr_per_frame*(j)
            tr_end=tr_per_frame*(j+1)
            ksp_ta=torch.from_numpy(ksp[:,tr_start:tr_end,:RO]).cuda()
            coord_t=torch.from_numpy(coord[tr_start:tr_end,:RO]).cuda()
            dcf_t=torch.from_numpy(dcf[tr_start:tr_end,:RO]).cuda()
            Pt=(P[:,tr_start:tr_end]).cuda()
           # from cupyx.scipy import ndimage
           # testing=cupyx.scipy.ndimage.map_coordinates(cupy.abs(cupy.asarray(im_test[0].detach().cpu().numpy())), cupy.asarray(new_locs.detach().cpu().numpy()), output=None, order=3, mode='reflect', cval=0.0, prefilter=True)


            if j>=T1 and j<T1+50:
               # im_out1=torch.nn.functional.interpolate(torch.abs(im_out), size=[210,123,219], scale_factor=None, mode='trilinear', align_corners=False, recompute_scale_factor=None) #+torch.nn.functional.interpolate(deform_fulla[j:j+1], size=[mps.shape[1],mps.shape[2],mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=None, recompute_scale_factor=None)
                deform_look[j-T1]=np.abs(flowa[:,0,:,:,28].detach().cpu().numpy())
                   # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
               # image_still[j-50]=np.abs(im_inv[0,0,:,:,35].detach().cpu().numpy())
                image_look[j-T1]=np.abs(im_out[:,:,28].detach().cpu().numpy())
                im_tester[j-T1]=np.squeeze(im_no_con[:,:,:].detach().cpu().numpy())
                con=torch.squeeze(con)
                im_test=torch.squeeze(im_test)
               # image_still[j-T1]=np.abs((con[:,:,120].detach().cpu().numpy())+(im_test[:,:,120].detach().cpu().numpy()))

            loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa.cuda())
            lo=0
            cooo=torch.ones([1])*lo
            loss_for=_updateb(im_out,ksp_ta,dcf_t,coord_t,mps) #+loss_grad+(torch.sum(deformL2a**2)+torch.sum(deformR2a**2))*1e-9+(torch.sum(deformL4a**2)+torch.sum(deformR4a**2))*1e-9+(torch.sum(deformL8a**2)+torch.sum(deformR8a**2))*1e-9
           #Q print(loss_for)
            loss_L=0
            loss_R=0
            loss_L0=0
            loss_R0=0
            for i in range(3):
              #  loss_L=loss_L+torch.norm(conL[i],'fro')**2
              #  loss_R=loss_R+torch.norm(conR[i][:,:,:,1:]-conR[i][:,:,:,:-1],'fro')**2
               
                loss_L0=loss_L0+torch.norm(deformL_param_adj[i],'fro')**2
                loss_R0=loss_R0+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
           # np.save('data/lung/deformL_param_adj0.npy',deformL_param_adj[0].detach().cpu().numpy())
           # np.save('data/lung/deformL_param_adj1.npy',deformL_param_adj[1].detach().cpu().numpy())
           # np.save('p1/deformL_param_adj2.npy',deformL_param_adj[2].detach().cpu().numpy())

          #  np.save('data/lung/deformR_param_adj0.npy',deformR_param_adj[0].detach().cpu().numpy())
          ##  np.save('data/lung/deformR_param_adj1.npy',deformR_param_adj[1].detach().cpu().numpy())
            #np.save('data/lung/deformR_param_adj2.npy',deformR_param_adj[2].detach().cpu().numpy())
           # count=count+1
           # torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
           # torch.nn.utils.clip_grad_value_([deform[i] for i in range(3)], 1e-1)
            print('loss')
            print(loss_for*weight_dc)
            print(loss_grad0*weight_smoother)
            print(loss_L*1e-5)
           # print(loss_rev)
            loss=loss_for*weight_dc/1+loss_grad0*weight_smoother/1 #+loss_R*1e-7 #+loss_L0*1e- #+loss_R0*1e-6+loss_L0*1e-6 #+loss_L0*1e-9+loss_R0*1e-7 #+loss_L*1e-8+loss_R*1e-8 #+loss_L*1e-6 #loss_grad0*weight_smoother #+loss_R*1e-4 #+loss_L*1e-8 #+loss_R*1e-6 #+loss_L*1e-9+loss_R*1e-9 #+loss_R*1e-6

            return loss
    count=0
    count0=0
    L=np.zeros([1])
    for io in range(iter_adj):
          #  optimizer0.zero_grad()

           # loss_grad0=0
           # loss_tot=0
           # loss_for=0
           # loss_rev=0
           # loss_for=0
           # loss_grad0=0
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])

         
            K=random.sample(range(T1,T2), T2-T1)
            print(K)
            for j in K:
                jo=j
                #j=91
                print(count)
               # print(j)
               
                count0=count%1
                L[count0]=np.int(j)
               # print(count)
               # print(j)
                               
              #  print(j)
               # optimizer0.zero_grad()


                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
               # con=cons(conL,conR,j-T1,block_torchcon,ishapecon)
               # print(con.shape)
               # all0=torch.abs(con).max()
              #  con_real=torch.squeeze(torch.nn.functional.interpolate(torch.real(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
              #  con_imag=torch.squeeze(torch.nn.functional.interpolate(torch.imag(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
               # con=torch.complex(con_real,con_imag)
              #  all1=torch.abs(con).max()
              
                flowa=deforma
                flowa=torch.nn.functional.interpolate(flowa, size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                print(flowa.shape)
                #flowa[:,0]=flowa[:,0]*inter
                #flowa[:,1]=flowa[:,1]*3
                #flowa[:,2]=flowa[:,2]*3
               # flowa_rev=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)

               # loss_for=0


                im_test=torch.from_numpy(im_template).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                con0=im_test #con
                im_out=warp1(con0,flowa,mps,complex=True)
                print('im_out')
                print(im_out.shape)
               # flowa_inv=inverse_field(flowa)
               # im_rev=warp0(im_out,flowa_inv,complex=True)
                
                tr_per_frame=spokes_per_bin
                tr_start=tr_per_frame*(j)
                tr_end=tr_per_frame*(j+1)
                ksp_ta=torch.from_numpy(ksp[:,tr_start:tr_end,:RO]).cuda()
                coord_t=torch.from_numpy(coord[tr_start:tr_end,:RO]).cuda()
                dcf_t=torch.from_numpy(dcf[tr_start:tr_end,:RO]).cuda()
                Pt=(P[:,tr_start:tr_end]).cuda()
               # from cupyx.scipy import ndimage
               # testing=cupyx.scipy.ndimage.map_coordinates(cupy.abs(cupy.asarray(im_test[0].detach().cpu().numpy())), cupy.asarray(new_locs.detach().cpu().numpy()), output=None, order=3, mode='reflect', cval=0.0, prefilter=True)

               # flowa=torch.squeeze(flowa)
                print(flowa.shape)
                if j>=T1 and j<T1+50:
                   # im_out1=torch.nn.functional.interpolate(torch.abs(im_out), size=[210,123,219], scale_factor=None, mode='trilinear', align_corners=False, recompute_scale_factor=None) #+torch.nn.functional.interpolate(deform_fulla[j:j+1], size=[mps.shape[1],mps.shape[2],mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=None, recompute_scale_factor=None)
                    deform_look[j-T1]=(flowa[:,0,:,40,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_inv[0,0,:,:,35].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_out[:,40,:].detach().cpu().numpy())
                 #   im_tester[j-T1]=np.squeeze(im_no_con[:,30,:].detach().cpu().numpy())
                   # con=torch.squeeze(con)
                    im_test=torch.squeeze(im_test)
                   # image_still[j-T1]=np.abs((con[:,:,120].detach().cpu().numpy())+(im_test[:,:,120].detach().cpu().numpy()))

                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa)
                lo=0
                cooo=torch.ones([1])*lo
                print('im_out')
                print(im_out.shape)
                print('mps')
                print(mps.shape)
                loss_for=torch.utils.checkpoint.checkpoint(_updateb,im_out.unsqueeze(0),ksp_ta,dcf_t,coord_t,mps) #+loss_grad+(torch.sum(deformL2a**2)+torch.sum(deformR2a**2))*1e-9+(torch.sum(deformL4a**2)+torch.sum(deformR4a**2))*1e-9+(torch.sum(deformL8a**2)+torch.sum(deformR8a**2))*1e-9
               #Q print(loss_for)
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
                for i in range(3):
                   # loss_L=loss_L+torch.norm(conL[i],'fro')**2
                   # loss_R=loss_R+torch.norm(conR[i][:,:,:,:])**2 #-conR[i][:,:,:,:-1],'fro')**2
                   
                    loss_L0=loss_L0+torch.norm(deformL_param_adj[i],'fro')**2
                    loss_R0=loss_R0+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
               # np.save('data/lung/deformL_param_adj0.npy',deformL_param_adj[0].detach().cpu().numpy())
               # np.save('data/lung/deformL_param_adj1.npy',deformL_param_adj[1].detach().cpu().numpy())
               # np.save('p1/deformL_param_adj2.npy',deformL_param_adj[2].detach().cpu().numpy())

              #  np.save('data/lung/deformR_param_adj0.npy',deformR_param_adj[0].detach().cpu().numpy())
              ##  np.save('data/lung/deformR_param_adj1.npy',deformR_param_adj[1].detach().cpu().numpy())
                #np.save('data/lung/deformR_param_adj2.npy',deformR_param_adj[2].detach().cpu().numpy())
                count=count+1
               # torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
               # torch.nn.utils.clip_grad_value_([deform[i] for i in range(3)], 1e-1)
                print('loss')
                print(loss_for*weight_dc)
                print(loss_grad0*weight_smoother)
                print(loss_L*1e-5)
               # print(loss_rev)
                loss=loss_for*weight_dc/1+loss_grad0*weight_smoother+loss_L0*1e-7+loss_R0*1e-6 #+loss_L0*1e-7+loss_R0*1e-5 #+loss_L*1e-5+loss_R*1e-5 #+loss_L0*1e-7+loss_R0*1e-7 #+loss_L0*1e-9+loss_R0*1e-7 #+loss_L*1e-8+loss_R*1e-8 #+loss_L*1e-6 #loss_grad0*weight_smoother #+loss_R*1e-4 #+loss_L*1e-8 #+loss_R*1e-6 #+loss_L*1e-9+loss_R*1e-9 #+loss_R*1e-6
                (loss).backward()
                #loss.backward()
                (optimizer0).step()
               # optimizer1.step()
                optimizer0.zero_grad()
              #  optimizer1.zero_grad()
               # scalar.update()
               
               
               # loss_for=0
               # loss_grad0=0
                   
           # loss_for=0
           #         loss_grad0=0
          #  scheduler.step()      
           # optimizer1.step(closure_adj)
            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=10)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=10)
            imageio.mimsave('image_tester.gif', [np.abs(im_tester[i,:,:])*1e15 for i in range(50)], fps=10)
           # imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(250)], fps=10)
    return deformL_param_adj,deformR_param_adj

def con_create(T,rank,block_size,scale,mps):

   # import cupy
    import numpy as np
    #L0=np.load('L0.npy','r+')
    #L1=np.load('L1.npy','r+')
    #L2=np.load('L2.npy','r+')
    #R0=np.load('R0.npy','r+')
    #R1=np.load('R1.npy','r+')
    #R2=np.load('R2.npy','r+')

    #L=[L0,L1,L2]
    #R=[R0,R1,R2]
    #mps=np.load('mps.npy','r+')
    #mps=np.zeros([150,38,39,89])
    import math
    import torch
    from math import ceil

    block_torch0=[]
    block_torch1=[]
    blockH_torch=[]
   
    ishape0a=[]
    ishape1a=[]
    j=0
 
    conL=[]
    conR=[]
    block_size0=[block_size]
    #gen



    #Ltorch=[]
    #Rtorch=[]
   # import torch_optimizer as optim


    for jo in block_size:

        b_j = [min(i, jo) for i in [mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale]]
        s_j = [(b+1)//2  for b in b_j]
        i_j = [ceil((i - b + s) / s) * s + b - s 
        for i, b, s in zip([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], b_j, s_j)]
        import sigpy as sp
        block=sp.linop.BlocksToArray(i_j, b_j, s_j)
        C_j = sp.linop.Resize([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], i_j,
                                      ishift=[0] * 3, oshift=[0] * 3)
       # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
        w_j = sp.hanning(b_j, dtype=np.float32, device=0)**0.5
        W_j = sp.linop.Multiply(block.ishape, w_j)
        block_final=C_j*block*W_j
        ishape1a.append(block_final.ishape)
        block_torch1.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
       
       # temp0_contrast_real=torch.ones([1,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
       # temp0_contrast_imag=torch.ones([1,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
       # temp0_contrast=torch.complex(temp0_contrast_real,temp0_contrast_imag)
       # temp0_contrast=1e2*temp0_contrast/torch.sum(torch.square(temp0_contrast))**0.5
        #print(torch.abs(temp0_contrast).max())
        #conL.append(torch.nn.parameter.Parameter(temp0_contrast,requires_grad=True))
        #conR.append(torch.nn.parameter.Parameter(torch.zeros([1,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda',dtype=torch.complex64),requires_grad=True))
      
    return block_torch1,ishape1a

def gen_con(block_torcha,conL,conR,ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo):
    j=int(jo[0])
   
    conR=conR.permute((1,2,3,4,5,6,0))
  
    conR0=torch.reshape(conR,[1,conR.shape[0]*conR.shape[1]*conR.shape[2],1,conR.shape[6]])
   # conL=conL.permute(3,4,5,0,1,2)
   
    conL0=torch.reshape(conL,[1,conL.shape[0]*conL.shape[1]*conL.shape[2],conL.shape[3]*conL.shape[4]*conL.shape[5],1])
    
    con=torch.matmul(conL0,conR0[:,:,:,j:j+1])
    con_real=torch.real(con)
    con_imag=torch.imag(con)
    con_real=torch.reshape(con_real,[int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    con_real=torch.squeeze(block_torcha.apply(con_real)).unsqueeze(0)
    con_imag=torch.reshape(con_imag,[int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    con_imag=torch.squeeze(block_torcha.apply(con_imag)).unsqueeze(0)
    con=torch.complex(con_real,con_imag)
    
   # deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
    return con
def cons(conL,conR,j,block_torch1,ishape1a):
        jo=torch.ones([1])*j
        con_rank=0
        #count=int(counta[0])
        for count in range(2):
           # print(count)
            ishape0=ishape1a[count][0]*torch.ones([1])
            ishape1=ishape1a[count][1]*torch.ones([1])
            ishape2=ishape1a[count][2]*torch.ones([1])
            ishape3=ishape1a[count][3]*torch.ones([1])
            ishape4=ishape1a[count][4]*torch.ones([1])
            ishape5=ishape1a[count][5]*torch.ones([1])
            con_num=torch.utils.checkpoint.checkpoint(gen_con,block_torch1[count],conL[count],conR[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo)
            con_rank=con_rank+con_num
            
        return con_rank
    
#to do: put in seperate .py

def select_res(ksp,dcf,coord,RO,spokes_per_bin):
    import sigpy as sp

    
   
    ksp=ksp[:,:,:RO]
    coord=coord[:,:RO]
    dcf=dcf[:,:RO]
    shape=sp.estimate_shape(coord)
    matrix_dim=np.ones([1,shape[0],shape[1],shape[2]])


    #generate sense maps
    import sigpy.mri as mr
    import sigpy as sp
    device=0
    mps = mr.app.JsenseRecon(ksp, coord=coord, weights=dcf, device=0).run()
    print(mps.shape)

    #normalize data
    device=0
    dcf=normalize(mps,coord,dcf,ksp,spokes_per_bin)
    ksp=kspace_scaling(mps,dcf,coord,ksp)
    T = 1
    device=0
    lamda = 1e-8
    blk_widths = [8,16,32]  # For low resolution.
    L_blocks,R_blocks,B,imga = MultiScaleLowRankRecon(ksp, coord, dcf, mps, T, lamda, device=device, blk_widths=blk_widths).run()

    mpsa=mps
    im_test=np.zeros([1,mpsa.shape[1],mpsa.shape[2],mpsa.shape[3]],dtype=np.complex64)
    temp=0
    for i in range(1):
        for j in range(3):
            temp=temp+B[j](L_blocks[j]*R_blocks[j][i])
        im_test[i]=temp.get()
    im_testa=im_test

    return ksp,coord,dcf,mps,matrix_dim,im_testa



def test_warp(im_testa,deformL_param_adj,deformR_param_adj,block_torch,ishape,T1,T2,interp):
    #from utils_reg1 import flows
    deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    #im_tester=np.zeros([T])
    for j in range(T1,T2):
                loss_rev1=0

                print(j)
                
               # optimizer0.zero_grad()


                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                flowa=deforma.cuda()
                new_res=im_testa
                flowa=torch.nn.functional.interpolate(flowa, size=[im_testa.shape[1],im_testa.shape[2],im_testa.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
               # deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                print(flowa.shape)
                deforma_inv=inverse_field(flowa)
                im_for=warp(im_testa,flowa)
                im_rev=warp(im_for,deforma_inv)
                '''
                deforma_inv=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)
                flowa_inv=deforma_inv.cuda()
                flowa_inv=torch.nn.functional.interpolate(flowa_inv, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
              
                loss_for=0


                im_test=(torch.from_numpy(im_testa).cuda().unsqueeze(0))
                all0=torch.abs(im_test).max()

                im_test=im_test

                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locs = new_locs[..., [2,1,0]]
                ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_out=torch.complex(ima_real,ima_imag)
              
               
                
                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [2,1,0]]
                im_inv_real=torch.nn.functional.grid_sample(torch.real(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv_imag=torch.nn.functional.grid_sample(torch.imag(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
                #im_inv=torch.nn.functional.grid_sample(im_out, new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                diff=im_inv-im_test
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
               # loss_self1=torch.nn.MSELoss()
                #loss_rev1=loss_self1(torch.squeeze(im_inv),torch.squeeze(torch.abs(im_test)))*10
                 '''

                if j>=T1 and j<T2:
                   # deform_look[j-T1]=np.abs(flowa_inv[0,:,:,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                    image_still[j-T1]=np.abs(im_inv[:,100,:].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_for[:,100,:].detach().cpu().numpy())
    import imageio
    imageio.mimsave('image_for.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(T2-T1)], fps=10)
   # imageio.mimsave('deform1.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(T2-T1)], fps=10)
    imageio.mimsave('image_inv.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(T2-T1)], fps=10)            
                   # image_still[j-50]=np.abs(im_test.detach().cpu().numpy())

    


def train_interp_field(im_template,new_res,old_res,deformL_param_adj0,deformR_param_adj0,deformL_param_adj,deformR_param_adj,deformL_param_for0,deformR_param_for0,deformL_param_for,deformR_param_for,iter,T,block_torch0,ishape0,block_torch1,ishape1,high_res_interp):
    scaler = torch.cuda.amp.GradScaler()
    optimizer=torch.optim.Adam([{'params':[deformL_param_adj[i] for i in range(3)],'lr':.001},{'params':[deformR_param_adj[i] for i in range(3)],'lr':.001},{'params':[deformL_param_for[i] for i in range(3)],'lr':.001},{'params':[deformR_param_for[i] for i in range(3)],'lr':.001}])
    '''
    code for interpolating low res to high res field in MSLR representation
    initial res uses dense deformation field, new_res uses control points
    variable definitions:
    deformL_param_adja/deformR_param_adja/deformL_param_fora/deformR_param_fora=old res variables
    blocktorch0/ishape0: old_res
    deformL_param_adj/deformR_param_adj/deformL_param_for/deformR_param_for=new res
    blocktorch1/ishape1=new_res
    iter: specify number of iterations
    T: number of frames
    '''
   
   # from utils_reg1 import flows
    image_look=np.zeros([50,new_res.shape[1],new_res.shape[3]])
    deform_look=np.zeros_like(image_look)
    #high_res_interp is scaling from control points in deformL to dense deformation field
    for i in range(iter):
        optimizer.zero_grad()

        loss_tot=0
        mps=new_res
        K=random.sample(range(0,T), T)
        for j in K:
          #  print(j)
            loss=0
           # print(j)
            #flowa=0

            #old_res-->new_res deformation field interpolation
            flowingc0=flows(deformL_param_adj0,deformR_param_adj0,j,block_torch0,ishape0)
            import cupyx
            from cupyx.scipy import ndimage
            flowingc=cupy.zeros([1,3,new_res.shape[1],new_res.shape[2],new_res.shape[3]])
            #rescaling for deformation fieldnterpolation
            scale0=new_res.shape[1]/old_res.shape[1]
            scale1=new_res.shape[2]/old_res.shape[2]
            scale2=new_res.shape[3]/old_res.shape[3]
            for i in range(3):
                flowingc[0,i]=cupyx.scipy.ndimage.zoom(cupy.asarray(flowingc0[0,i]), [scale0,scale1,scale2], output=None, order=3, mode='reflect', cval=0.0, prefilter=True, grid_mode=True) #interpolate using splines
           # flowingc=torch.nn.functional.interpolate(flowingc, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
            flowingc=torch.as_tensor(flowingc,device='cuda')
            flowingc=flowingc.float()
            flowingc[:,0]=flowingc[:,0]*new_res.shape[1]/old_res.shape[1]
            flowingc[:,1]=flowingc[:,1]*new_res.shape[2]/old_res.shape[2]
            flowingc[:,2]=flowingc[:,2]*new_res.shape[3]/old_res.shape[3]
            
            #new_res deformation field triilnear interpolation from control points
            flowb=flows(deformL_param_adj,deformR_param_adj,j,block_torch1,ishape1)
            flowa=torch.nn.functional.interpolate(flowb, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
            flowa[:,0]=flowa[:,0]*high_res_interp
            flowa[:,1]=flowa[:,1]*high_res_interp
            flowa[:,2]=flowa[:,2]*high_res_interp
            '''
            #same logic for inv_field
            flowingc_inv0=flows(deformL_param_for0,deformR_param_for0,j,block_torch0,ishape0)
            flowingc_inv=cupy.zeros([1,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            scale0=new_res.shape[1]/old_res.shape[1]
            scale1=new_res.shape[2]/old_res.shape[2]
            scale2=new_res.shape[3]/old_res.shape[3]
            for i in range(3):
                flowingc_inv[0,i]=cupyx.scipy.ndimage.zoom(cupy.asarray(flowingc_inv0[0,i]), [scale0,scale1,scale2], output=None, order=3, mode='reflect', cval=0.0, prefilter=True, grid_mode=True)
            #flowingc_inv=torch.nn.functional.interpolate(flowingc_inv, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
            flowingc_inv=torch.as_tensor(flowingc_inv,device='cuda') 
            flowingc_inv=flowingc_inv.float()
            flowingc_inv[:,0]=flowingc_inv[:,0]*new_res.shape[1]/old_res.shape[1]
            flowingc_inv[:,1]=flowingc_inv[:,1]*new_res.shape[2]/old_res.shape[2]
            flowingc_inv[:,2]=flowingc_inv[:,2]*new_res.shape[3]/old_res.shape[3]
            flowb_inv=flows(deformL_param_for,deformR_param_for,j,block_torch1,ishape1)
            flowa_inv=torch.nn.functional.interpolate(flowb_inv, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
            flowa_inv[:,0]=flowa_inv[:,0]*high_res_interp
            flowa_inv[:,1]=flowa_inv[:,1]*high_res_interp
            flowa_inv[:,2]=flowa_inv[:,2]*high_res_interp
            '''
            loss_self1=torch.nn.MSELoss()
           # loss_inv_smooth=f.loss(flowa_inv)*1000
           # loss_for_smooth=f.loss(flowa)*10
            loss_flow_for=loss_self1(flowa[:,:],flowingc)*1
           # loss_flow_inv=loss_self1(flowa_inv,flowingc_inv)
           # loss_flow_inv=loss_self1(flowa_inv[:,:],flowingc_inv)*100
           # im_test=(torch.from_numpy(im_testa).unsqueeze(0))
           # all0=torch.abs(im_test).max()

            im_test=torch.from_numpy(im_template).cuda()
            
            with torch.no_grad():
          #  spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locs = new_locs[..., [2,1,0]]
                im_test=im_test.unsqueeze(0)
                ima_real=torch.nn.functional.grid_sample(torch.real(im_test).cuda(), new_locs.cuda(), mode='bilinear', padding_mode='zeros', align_corners=True)
                ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test).cuda(), new_locs.cuda(), mode='bilinear', padding_mode='zeros', align_corners=True)
                im_out=torch.complex(ima_real,ima_imag)
           
           
           # print(loss_inv_smooth)
           # print(loss_for_smooth)
            print(loss_flow_for)
           # print(loss_flow_inv)
            
            loss=loss_flow_for*1 #+loss_flow_inv*1 #    +loss_for_smooth #+loss_inv_smooth#+loss_self1(flowa[:,1:2].cuda(),flowingc.cuda())*100+loss_self1(flowa[:,2:3].cuda(),flowingc.cuda())*100
 
            (loss).backward()
            optimizer.step()
            optimizer.zero_grad()
           # loss=loss.detach().cpu().numpy()
         
            torch.cuda.empty_cache()
           # loss_tot=loss+loss_tot

            print('loss')
            print(loss)
            im_out=torch.squeeze(torch.abs(im_out))
            print(im_out.shape)
            if j>=0 and j<50:
                image_look[j]=im_out[:,110,:].detach().cpu().numpy()
                deform_look[j]=torch.squeeze(flowa[0,0,:,100,:]).detach().cpu().numpy()
    
            #optimizer.step()
            #optimizer.zero_grad()
        import imageio
        imageio.mimsave('image1.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=10)
        imageio.mimsave('deform1.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=10)
    return deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for



               
def for_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_testa,mps,iter,block_torch,ishape,T1,T2,interp,new_res,weight_MSE=1e-1):
     #from utils_reg1 import flows,_updateb,f
     from torch.utils import checkpoint
     import torch
     import numpy as np
     import random
     deform=[deformL_param_for[0],deformL_param_for[1],deformL_param_for[2],deformR_param_for[0],deformR_param_for[1],deformR_param_for[2]]
     optimizer2=torch.optim.Adam([deform[i] for i in range(6)],lr=.01) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 

   #  optimizer3=torch.optim.LBFGS([deformR_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 
     deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     P=torch.ones([40,1])
     mps=torch.from_numpy(mps).cpu()
     def closure_for():
       # from utils_reg1 import flows,_updateb,f
        from torch.utils import checkpoint
        import torch
        import numpy as np
        import random
        with torch.no_grad():
            loss_for=0
            loss_grad0=0
            loss_det=0
            loss_rev1=0

            for j in K:
       
                count=0
                lossz=np.zeros([20])

             
                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                
              

                flowa=deforma.cuda()
                new_res=mps
                flowa=torch.nn.functional.interpolate(flowa.unsqueeze(0), size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                flowa_inv=inverse_field(flowa,mps)
               # deforma_inv=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)
                flowa_inv=deforma_inv.cuda()
                flowa_inv=torch.nn.functional.interpolate(flowa_inv.unsqueeze(0), size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
              
                
                im_test=(torch.from_numpy(im_testa).cuda().unsqueeze(0)) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                im_test=im_test
               
              #  spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                vectors=[]
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors = [ torch.arange(0, s) for s in size ] 
                #vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                #new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locsa = new_locs[..., [0,1,2]]
               # ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
               # ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                ima_real=grid_pull(torch.squeeze(torch.real(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                ima_imag=grid_pull(torch.squeeze(torch.imag(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
              
               
              #  spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [0,1,2]]
                im_inv_real=grid_pull(torch.squeeze(torch.real(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv_imag=grid_pull(torch.squeeze(torch.imag(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
               # im_inv_real=torch.nn.functional.grid_sample(torch.real(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
               # im_inv_imag=torch.nn.functional.grid_sample(torch.imag(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
           
                diff=im_inv-im_test
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
              
                loss_grad0=loss_grad0+torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())
            
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
               # for i in range(5):
                 #   loss_L=loss_L+torch.norm(deformL_param_adj[i],'fro')**2
                 #   loss_R=loss_R+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
               # loss_L0=loss_L0+torch.norm(Ltorch[i],'fro')**2
               # loss_R0=loss_R0+torch.norm(Rtorch[i],'fro')**2
      
            loss=loss_rev1*weight_MSE+loss_grad0 #+loss_grad0*30 #+loss_R*1e-6 #loss_L0*1e-8+loss_R0*1e-8 
            return loss

     for io in range(iter):

            
            optimizer2.zero_grad()
          #  optimizer3.zero_grad()
       
            loss_grad0=0
            loss_tot=0
            loss_for=0
            loss_rev=0
            loss_for=0
            loss_grad0=0
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])
          
            K=random.sample(range(T1,T2), T2-T1)
            for j in K:
                loss_rev1=0

                print(j)
               # optimizer0.zero_grad()


                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                print(deforma.shape)
                flowa=deforma.cuda()
                new_res=mps
                flowa=torch.nn.functional.interpolate(flowa, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                #flowa_inv=inverse_field(flowa,mps)
                deforma_inv=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)
                flowa_inv=deforma_inv.cuda()
                flowa_inv=torch.nn.functional.interpolate(flowa_inv.unsqueeze(0), size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
               
                loss_for=0


                im_test=(torch.from_numpy(im_testa).cuda().unsqueeze(0))
                all0=torch.abs(im_test).max()

                im_test=im_test

                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locsa = new_locs[..., [0,1,2]]
                ima_real=grid_pull(torch.squeeze(torch.real(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                ima_imag=grid_pull(torch.squeeze(torch.imag(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
               # ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
               # ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
               
                #ima=torch.nn.functional.grid_sample(im_test, new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
              

                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               ## for i in range(len(shape)):
                    #new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [0,1,2]]
                im_inv_real=grid_pull(torch.squeeze(torch.real(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv_imag=grid_pull(torch.squeeze(torch.imag(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
                #im_inv=torch.nn.functional.grid_sample(im_out, new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                diff=im_inv-im_test
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
               # loss_self1=torch.nn.MSELoss()
                #loss_rev1=loss_self1(torch.squeeze(im_inv),torch.squeeze(torch.abs(im_test)))*10


                if j>=T1 and j<T2+500:
                    deform_look[j-T1]=np.abs(flowa_inv[:,0,:,100,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach8.cpu().numpy())
                    image_still[j-T1]=np.abs(im_inv[:,100,:].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_out[:,100,:].detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_test.detach().cpu().numpy())

                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())

                lo=0
                print(loss_rev1*.5)
                print(loss_grad0*.5)
                
                loss=loss_rev1*2+loss_grad0*.5
                print(loss)

                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()



          #  optimizer3.step(closure_for)

            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(450)], fps=20)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(450)], fps=20)
            imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(450)], fps=20)
     return deformL_param_for,deformR_param_for


def gen_MSLR(T,rank,block_size_adj,block_size_for,scale,mps):

    import cupy
    import numpy as np
    
    import math
    import torch
    from math import ceil

    block_torch0=[]
    block_torch1=[]
    blockH_torch=[]
    #deformL_adj=[]
    #deformR_adj=[]
    deformL_for=[]
    deformR_for=[]
    ishape0a=[]
    ishape1a=[]
    j=0
    deformL_param_adj=[]
    deformR_param_adj=[]
    deformL_param_for=[]
    deformR_param_for=[]
    #gen


    block_size0=block_size_adj
    block_size1=block_size_for

    #Ltorch=[]
    #Rtorch=[]
    import torch_optimizer as optim


    for jo in block_size0:
        print(jo)

        b_j = [min(i, jo) for i in [mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale]]
        print(b_j)
        s_j = [(b+1)//2  for b in b_j]
        i_j = [ceil((i - b + s) / s) * s + b - s 
        for i, b, s in zip([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], b_j, s_j)]
        import sigpy as sp
        block=sp.linop.BlocksToArray(i_j, b_j, s_j)
       # print(block.shape)
        C_j = sp.linop.Resize([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], i_j,
                                      ishift=[0] * 3, oshift=[0] * 3)
       # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
        w_j = sp.hanning(b_j, dtype=cupy.float32, device=0)**0.5
        W_j = sp.linop.Multiply(block.ishape, w_j)
        block_final=C_j*block*W_j
        ishape1a.append(block_final.ishape)
        block_torch1.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
       
        temp0=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
  
        temp0=1e3*temp0/torch.sum(torch.square(torch.abs(temp0)))**0.5
        print(temp0.max())
        deformL_param_adj.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
       # tempa=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],dtype=torch.float16,device='cuda')
        deformR_param_adj.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
        deformL_param_for.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
        deformR_param_for.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
    return deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,block_torch1,ishape1a

def gen(block_torcha,deformL_param,deformR_param,ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo):
    jb=int(jo[0])
   # print(jb)
    deform_patch_adj=torch.matmul(deformL_param,deformR_param[:,:,:,jb:jb+1])
    deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    deformx_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[0])).unsqueeze(0)
    deformy_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[1])).unsqueeze(0)
    deformz_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[2])).unsqueeze(0)
   # deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
    return deformx_adj,deformy_adj,deformz_adj

def flows(deformL_param_adj,deformR_param_adj,j,block_torch1,ishape1a):
        jo=torch.ones([1])*j
        deform_adj=[]
        deform_for=[]
        #count=int(counta[0])
        for count in range(3):
           # print(count)
            ishape0=ishape1a[count][0]*torch.ones([1])
            ishape1=ishape1a[count][1]*torch.ones([1])
            ishape2=ishape1a[count][2]*torch.ones([1])
            ishape3=ishape1a[count][3]*torch.ones([1])
            ishape4=ishape1a[count][4]*torch.ones([1])
            ishape5=ishape1a[count][5]*torch.ones([1])
       # deformx0,deformy1,deformz0=torch.utils.checkpoint.checkpoint(gen,block_torch0,deformL_param_adj0,deformR_param_adj0,ishape00,ishape10,ishape20,ishape30,ishape40,ishape50,jo)
            deformx,deformy,deformz=gen(block_torch1[count],deformL_param_adj[count],deformR_param_adj[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo)
           # deform_for.append(torch.cat([deformx,deformy,deformz],axis=0))
           # deformx,deformy,deformz=torch.utils.checkpoint.checkpoint(gen,block_torch[count],deformL_param_for[count],deformR_param_for[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo,preserve_rng_state=False)
            deform_adj.append(torch.cat([deformx,deformy,deformz],axis=0))
        flow=deform_adj[0]+deform_adj[1]+deform_adj[2] #+deform_adj[2] #+deform_adj[3] #+deform_adj[4] #+deform_adj[3]+deform_adj[4]+deform_adj[5] #+deform_adj[6]+deform_adj[7]
        flow=flow.unsqueeze(0)
        return flow
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) #*w0
       
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) #*w1
      
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) #*w2
       # dt = torch.abs(y_pred[1:, :, :, :, :] - y_pred[:-1, :, :, :, :])

      
        dy = dy
        dx = dx
        dz = dz
            #dt=dt*dt

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

     
        return grad
    
f=Grad()

def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf):
        torch.cuda.empty_cache()
        loss=0
      
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(),1.25,2)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t.cuda())
       # print(e_tc.shape)
        #Ptc=torch.reshape(Ptc,[-1])
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1]) #*torch.reshape(Ptc,[-1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
      #  print(torch.abs(e_tca).max())
      #  print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        #ksp_real=torch.real(ksp)*dcf
        #ksp_imag=torch.imag(ksp)*dcf
        #e_tca_real=torch.real(e_tca)*dcf
        #e_tca_imag=torch.imag(e_tca)*dcf
       # ksp=torch.complex(ksp_real,ksp_imag)
       # e_tca=torch.complex(e_tca_real,e_tca_imag)
        res=(ksp-e_tca)*(dcf)**0.5 #**0.5 #**0.5
        res=torch.reshape(res,[1,-1])
        lossb=(torch.linalg.norm(res,2))**2 #**2 #**2 #torch.abs(torch.sum(((ksp-e_tca))))
        
        #loss=(torch.norm(resk)) #/torch.norm(ksp,2)+torch.norm(resk,1)/torch.norm(ksp,1) #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
       
        
       # torch.cuda.empty_cache()
       # loss=torch.norm(resk,1) #*index_all/index_max  #*index_all/ksp.shape[0]
        #print(loss)
        return lossb

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  #print(mpsa.shape)
  #print(img_t.shape)
  #n=int(cooo[0])
  for c in range(mpsa.shape[0]):
       
    
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,torch.cat([torch.reshape(torch.real(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1]),torch.reshape(torch.imag(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mpsa[c],torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
   # m=mpsa[c].detach().cpu().numpy()
   # torch.cuda.empty_cache()
  #del img_t
  #torch.cuda.empty_cache()
  
  return loss_t

'''
for jo in block_size:
    b_j = [min(i, jo) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
    s_j = [(b+1)//2  for b in b_j]
    i_j = [ceil((i - b + s) / s) * s + b - s 
    for i, b, s in zip([mps.shape[1],mps.shape[2],mps.shape[3]], b_j, s_j)]
    import sigpy as sp
    block=sp.linop.BlocksToArray(i_j, b_j, s_j)
    C_j = sp.linop.Resize([mps.shape[1],mps.shape[2],mps.shape[3]], i_j,
                                  ishift=[0] * 3, oshift=[0] * 3)
   # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
    w_j = sp.hanning(b_j, dtype=cupy.float32, device=0)**0.5
    W_j = sp.linop.Multiply(block.ishape, w_j)
    block_final=C_j*block*W_j
    blocka.append(block_final)
    ishape1p.append(block_final.ishape)
    tempa=torch.rand([1,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*10
    #deformR_adj.append(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'))
    
    Lmov.append(tempa)
    Rmov.append(torch.rand([1,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda')*.1)
    #ishape.append(block_final.ishape)
    blocka_torch.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
   # blockHa_torch.append(sp.to_pytorch_function(block_final.H,input_iscomplex=False,output_iscomplex=False))
'''
Lprime_param=[]
Rprime_param=[]
#deformL_param_adj=[]
#deformR_param_adj=[]
#for j in range(3):
#    print(j)
#    deformL_param_adj.append(torch.nn.parameter.Parameter(deformL_adj[j],requires_grad=True))
#    deformR_param_adj.append(torch.nn.parameter.Parameter(deformR_adj[j],requires_grad=True))  
   # Lprime_param.append(torch.nn.parameter.Parameter((Lmov[j]),requires_grad=True))
   # Rprime_param.append(torch.nn.parameter.Parameter((Rmov[j]),requires_grad=True)) 
#torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#import torch_optimizer as optim
#optimizer=optim.AdaBound([{'params':[deformL_param_adj[i] for i in range(3)],'lr':1e-2,'eps':1e-8,'betas':(0.9, 0.999)},{'params':[deformR_param_adj[i] for i in range(3)],'lr':1e-1,'eps':1e-8,'betas':(0.9, 0.999)}])
    
    #Asabound
#generates images from L and R bases (for pairwise registrations allows us to work entirely in compressed space)

def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()
    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=True)
import argparse
import logging

import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
#from multi_scale_low_rank_image import MultiScaleLowRankImage
import numpy as np
#import cupy as xp
import cupy as xp
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

#import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
#from utils_reg1 import _updateb,f
#from utils.misc import param_ndim_setup

import os
import random
from typing import Dict, Any

  
def normalize(mps,coord,dcf,ksp,tr_per_frame):
    mps=mps
   
   # import cupy
    device=0
    import sigpy as sp
 
    # Estimate maximum eigenvalue.
    coord_t = sp.to_device(coord[:tr_per_frame], device)
    dcf_t = sp.to_device(dcf[:tr_per_frame], device)
    F = sp.linop.NUFFT([mps.shape[1],mps.shape[2],mps.shape[3]], coord_t)
    W = sp.linop.Multiply(F.oshape, dcf_t)

    max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500, device=0,
                            dtype=ksp.dtype,show_pbar=True).run()
    dcf1=dcf/max_eig
    return dcf1

def kspace_scaling(mps,dcf,coord,ksp):
    # Estimate scaling.
    img_adj = 0
    device=0
    dcf = sp.to_device(dcf, device)
    coord = sp.to_device(coord, device)
   
    for c in range(mps.shape[0]):
        print(c)
        mps_c = sp.to_device(mps[c], device)
        ksp_c = sp.to_device(ksp[c], device)
        img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, [mps.shape[1],mps.shape[2],mps.shape[3]])
        img_adj_c *= cupy.conj(mps_c)
        img_adj += img_adj_c


    img_adj_norm = cupy.linalg.norm(img_adj).item()
    print(img_adj_norm)
    ksp1=ksp/img_adj_norm
    return ksp1
import pathlib
import torch.nn.functional as F
from tqdm.auto import tqdm
import requests
import pathlib
#download_lung_dataset()
import argparse
import logging

import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
#from multi_scale_low_rank_image import MultiScaleLowRankImage
import numpy as np
#import cupy as xp
import cupy as xp
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

#import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
#from utils_reg1 import _updateb,f
#from utils.misc import param_ndim_setup

import os
import random
from typing import Dict, Any
from interpol import grid_pull

#import omegaconf
#be


def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.
    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension
    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param

class _Transform(object):
    """ Transformation base class """
    def __init__(self,
                 svf=True,
                 svf_steps=7,
                 svf_scale=1):
        self.svf = svf
        self.svf_steps = svf_steps
        self.svf_scale = svf_scale

    def compute_flow(self, x):
        raise NotImplementedError

    def __call__(self, x):
        flow = self.compute_flow(x)
        if self.svf:
            disp = svf_exp(flow,
                           scale=self.svf_scale,
                           steps=self.svf_steps)
            disp_inv=svf_exp_inv(flow, scale=self.svf_scale, steps=self.svf_steps, sampling='bilinear')
            return flow,disp,disp_inv
        else:
            disp = flow
            return disp



class DenseTransform(_Transform):
    """ Dense field transformation """
    def __init__(self,
                 svf=True,
                 svf_steps=7,
                 svf_scale=1):
        super(DenseTransform, self).__init__(svf=svf,
                                             svf_steps=svf_steps,
                                             svf_scale=svf_scale)

    def compute_flow(self, x):
        return x


class CubicBSplineFFDTransform(_Transform):
    def __init__(self,
                 ndim,
                 img_size=192,
                 cps=5,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.
        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            svf: (bool) stationary velocity field formulation if True
        """
        super(CubicBSplineFFDTransform, self).__init__(svf=svf,
                                                       svf_steps=svf_steps,
                                                       svf_scale=svf_scale)
        self.ndim = ndim
        self.img_size = param_ndim_setup(img_size, self.ndim)
        self.stride = param_ndim_setup(cps, self.ndim)

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2
                        for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def compute_flow(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            y: (N, dim, *(img_sizes)) The dense flow field of the transformation
        """
        # separable 1d transposed convolution
        flow = x
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]
        return flow


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.
    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field
    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def svf_exp(flow, scale=1, steps=5, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def svf_exp_inv(flow, scale=1, steps=5, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp - warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline1d(stride, derivative: int = 0, dtype=None, device= None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.
    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.
    Returns:
        Cubic B-spline convolution kernel.
    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def conv1d(
        data: Tensor,
        kernel: Tensor,
        dim: int = -1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        transpose: bool = False
) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    result = data.type(kernel.dtype)  # (n, ndim, h, w, d)
    result = result.transpose(dim, -1)  # (n, ndim, ..., shape[dim])
    shape_ = result.size()
    # use native pytorch
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    # groups = numel(shape_[1:-1])  # (n, nidim * shape[not dim], shape[dim])
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # 3*w*d, 1, kernel_size
    result = result.reshape(shape_[0], groups, shape_[-1])  # n, 3*w*d, shape[dim]
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(
        result,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = result.transpose(-1, dim)
    return result


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()
    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=True)


class MultiScaleLowRankRecon:
    r"""Multi-scale low rank reconstruction.
    Considers the objective function,
    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)
    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.
    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.
    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda, ishape,deformL_param_adj , deformR_param_adj,deformL_param_for,deformR_param_for,block_torch,
                 blk_widths=[32, 64, 128], alpha=.1, beta=.5,sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=60, decay_epoch=20, max_power_iter=1,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)
        self.scale=1
        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
       
       
        self.deform_look=np.zeros([50,mps.shape[1],mps.shape[3]])
        self.max_epoch = max_epoch
        self.block_torch=block_torch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)
        self.ishape=ishape
        self.deformL_param_adj=deformL_param_adj
        self.deformR_param_adj=deformR_param_adj
        self.deformL_param_for=deformL_param_for
        self.deformR_param_for=deformR_param_for
       # self.deform.requires_grad=True
       # self.deform_adjoint.requires_grad=True
       # optimizer = torch.optim.Adam([
     #{'params':self.deform_adjoint,'lr':.1}])
      #  self.optimizer=optimizer

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

       # self._normalize()
    
    def get_adj_field(self,t):
         flow=0
         mps=self.mps
         deform_adj=[]
         for count in range(3):
            #self.scale=3
            deform_patch_adj=torch.matmul(self.deformL_param_adj[count].cuda(),self.deformR_param_adj[count][:,:,:,t:t+1].cuda())
            deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[0])).unsqueeze(0)
            deformy_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[1])).unsqueeze(0)
            deformz_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[2])).unsqueeze(0)
            deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
            flow=flow+deform_adj[count] 
         #print(flow.shape)
         #flow=flow.unsqueeze(0)
         #flow=Bspline.compute_flow(flow)
         self.scale=3
         flow=torch.reshape(flow,[1,3,self.mps.shape[1]//self.scale,self.mps.shape[2]//self.scale,self.mps.shape[3]//self.scale])
        
         flow=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*3
         flow_adj=torch.reshape(flow,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
         
          
         return flow_adj
    def get_for_field(self,t):
        flow=0
        mps=self.mps
        deform_for=[]
        for count in range(3):
            deform_patch_for=torch.matmul(self.deformL_param_for[count].cuda(),self.deformR_param_for[count][:,:,:,t:t+1].cuda())
            deform_patch_for=torch.reshape(deform_patch_for,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[0])).unsqueeze(0)
            deformy_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[1])).unsqueeze(0)
            deformz_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[2])).unsqueeze(0)
            deform_for.append(torch.cat([deformx_for,deformy_for,deformz_for],axis=0))
            flow=flow+deform_for[count] 
         #flow=flow.unsqueeze(0)
         #flow=Bspline.compute_flow(flow)
        self.scale=3
        flow=torch.reshape(flow,[1,3,self.mps.shape[1]//self.scale,self.mps.shape[2]//self.scale,self.mps.shape[3]//self.scale])
        
        #flow=torch.reshape(flow,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        #scale=1
        flow_for=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*3
        flow_for=torch.reshape(flow_for,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        
        return flow_for
    
    def warp(self,flow,img):
        img=img.cuda()
        mps=self.mps
        img=torch.reshape(img,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
         
        spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
        size=shape
        vectors=[]
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        new_locs=grid.cuda()+flow
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
      #  for i in range(len(shape)):
      #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1) 
       # new_locs = new_locs[..., [2,1,0]]
        new_locsa = new_locs[..., [0,1,2]]

        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
         #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
         #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
        im_out=torch.complex(ima_real,ima_imag)
        im_out=torch.squeeze(im_out)
    


        
        return im_out
        

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue.
            coord_t = sp.to_device(self.coord[:self.tr_per_frame], self.device)
            dcf_t = sp.to_device(self.dcf[:self.tr_per_frame], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            self.dcf /= max_eig

            # Estimate scaling.
            img_adj = 0
            dcf = sp.to_device(self.dcf, self.device)
            coord = sp.to_device(self.coord, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=(-3,-2,-1), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            self.sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                self.sigma.append(L_j_norm)

        for j in range(self.J):
            self.L[j] *= self.sigma[j]**0.5
            self.R[j] *= self.sigma[j]**0.5
            
    def _AHyH_L(self, t):
        #t=0
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc
       # flow_for=self.get_for_field(t)
       # AHy_t=cupy.array(self.warp(flow_for,torch.as_tensor(AHy_tc)).detach().cpu().numpy())

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

    def _AHy_R(self, t):
       # ta=0
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc
       # flow_for=self.get_for_field(t)
       # AHy_t=cupy.array(self.warp(flow_for,torch.as_tensor(AHy_tc)).detach().cpu().numpy())

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])


    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                   
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                   
                   
                 
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return self.L,self.R
    def adj_field_solver(self,inner_iter,RO,T1,T2):
        from utils_reg1 import flows,_updateb,f
        from torch.utils import checkpoint
        import torch
        import numpy as np
        import random
        deform_look=np.zeros([T2-T1,self.mps.shape[1],self.mps.shape[2]])
        image_still=np.zeros([T2-T1,self.mps.shape[1],self.mps.shape[2]])
        image_look=np.zeros([T2-T1,self.mps.shape[1],self.mps.shape[2]])
        P=torch.ones([40,1])
        mps=torch.from_numpy(self.mps).cpu()
        for i in range(3):
            self.deformL_param_adj[i]=torch.nn.parameter.Parameter(self.deformL_param_adj[i],requires_grad=True)
            self.deformR_param_adj[i]=torch.nn.parameter.Parameter(self.deformR_param_adj[i],requires_grad=True)
        optimizer0=torch.optim.LBFGS([self.deformL_param_adj[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)

        optimizer1=torch.optim.LBFGS([self.deformR_param_adj[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 
       # optimizer3=torch.optim.LBFGS([deformL_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)

       # optimizer4=torch.optim.LBFGS([deformR_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 
        def closure_adj():
            from utils_reg1 import flows,_updateb,f
            from torch.utils import checkpoint
            import torch
            import numpy as np
            import random
            P=torch.ones([40,1])
            with torch.no_grad():
                loss_for=0
                loss_grad0=0
                loss_det=0

                for t in K:

                    count=0
                    lossz=np.zeros([20])

                    loss_rev=0

                    flowa=self.get_adj_field(t) #flows(deformL_param_adj,deformR_param_adj,j-50,block_torch,ishape)

                 # 
                  #  print(j)
                  #  if j==0:
                    im_test = 0
                    for j in range(self.J):
                        im_test += self.B[j](self.L[j] * self.R[j][t])
                    im_test=torch.as_tensor(im_test,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                    all0=torch.abs(im_test).max()
                    im_test=im_test
                    im_test=torch.reshape(im_test,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])

                    vectors=[]
                    shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
                    size=shape
                    vectors = [ torch.arange(0, s) for s in size ] 
                    #vectors = [ torch.arange(0, s) for s in size ] 
                    grids = torch.meshgrid(vectors) 
                    grid  = torch.stack(grids) # y, x, z
                    grid  = torch.unsqueeze(grid, 0)  #add batch
                    grid = grid.type(torch.FloatTensor)
                    new_locs=grid.cuda()+flowa.cuda()
                    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                    for i in range(len(shape)):
                        new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                    new_locs = new_locs[..., [2,1,0]]
                    ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                    ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)

                    im_out=torch.complex(ima_real,ima_imag)


                    tr_per_frame=self.ksp.shape[1]//self.T
                    print(tr_per_frame)
                    tr_start=tr_per_frame*(t)
                    tr_end=tr_per_frame*(t+1)


                   # with torch.no_grad():
                    ksp_ta=torch.from_numpy(self.ksp[:,tr_start:tr_end,:RO]).cuda()
                    coord_t=torch.from_numpy(self.coord[tr_start:tr_end,:RO]).cuda()
                    dcf_t=torch.from_numpy(self.dcf[tr_start:tr_end,:RO]).cuda()
                    Pt=(P[:,tr_start:tr_end]).cuda()

                    #torch.from_numpy(ksp[:,tr_start:tr_end,:RO]).cuda()

                       # flowa=torch.reshape(flowa,[210,123,219,3])
                    w0=torch.ones([1])
                    loss_grad0=loss_grad0+torch.utils.checkpoint.checkpoint(f.loss,flowa.cuda())
                    lo=0
                    cooo=torch.ones([1])*lo

                    loss_for=loss_for+torch.utils.checkpoint.checkpoint(_updateb,im_out,ksp_ta,dcf_t,coord_t,mps,cooo,Pt,preserve_rng_state=False)                                                              
                    loss_R=0
                    loss_L=0
                    loss_L0=0
                    loss_R0=0

                loss=(loss_for)*2+loss_grad0*30 #+loss_R*1e-6 #loss_L0*1e-8+loss_R0*1e-8 


                return loss
        for io in range(inner_iter):

                optimizer0.zero_grad()
                optimizer1.zero_grad()

                loss_grad0=0
                loss_tot=0
                loss_for=0
                loss_rev=0
                loss_for=0
                loss_grad0=0
                flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
                count=0
                lossz=np.zeros([20])

                K=random.sample(range(T1,T2), T2-T1)
                for t in K:

                    print(t)
                   # optimizer0.zero_grad()



                    flowa=self.get_adj_field(t) #flows(deformL_param_adj,deformR_param_adj,j-50,block_torch,ishape)

                 # 
                   # print(j)
                  #  if j==0:
                    im_test = 0
                    for j in range(self.J):
                        im_test += self.B[j](self.L[j] * self.R[j][t])
                    im_test=torch.as_tensor(im_test,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                    all0=torch.abs(im_test).max()
                    im_test=im_test
                    im_test=torch.reshape(im_test,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
                 
                    shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
                    size=shape
                    vectors=[]
                    vectors = [ torch.arange(0, s) for s in size ] 
                    grids = torch.meshgrid(vectors) 
                    grid  = torch.stack(grids) # y, x, z
                    grid  = torch.unsqueeze(grid, 0)  #add batch
                    grid = grid.type(torch.FloatTensor)
                    new_locs=grid.cuda()+flowa.cuda()
                    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                    for i in range(len(shape)):
                        new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                    new_locs = new_locs[..., [2,1,0]]
                    ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                    ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                    im_out=torch.complex(ima_real,ima_imag)
                    tr_per_frame=self.ksp.shape[1]//self.T
                    tr_start=tr_per_frame*(t)
                    tr_end=tr_per_frame*(t+1)
                    ksp_ta=torch.from_numpy(self.ksp[:,tr_start:tr_end,:RO]).cuda()
                    coord_t=torch.from_numpy(self.coord[tr_start:tr_end,:RO]).cuda()
                    dcf_t=torch.from_numpy(self.dcf[tr_start:tr_end,:RO]).cuda()
                    Pt=(P[:,tr_start:tr_end]).cuda()
                  
                   # testing=cupyx.scipy.ndimage.map_coordinates(cupy.abs(cupy.asarray(im_test[0].detach().cpu().numpy())), cupy.asarray(new_locs.detach().cpu().numpy()), output=None, order=3, mode='reflect', cval=0.0, prefilter=True)


                    if t>=T1 and t<T2:
                        deform_look[t-T1]=np.abs(flowa[:,0,:,:,80].detach().cpu().numpy())
                           # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                       # image_still[j-50]=np.abs(im_inv[0,0,:,:,35].detach().cpu().numpy())
                        image_look[t-T1]=np.abs(im_out[0,0,:,:,80].detach().cpu().numpy())
                       # image_still[j-50]=np.abs(im_test.detach().cpu().numpy())

                    loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa.cuda())
                    lo=0
                    cooo=torch.ones([1])*lo
                    loss_for=torch.utils.checkpoint.checkpoint(_updateb,im_out,ksp_ta,dcf_t,coord_t,mps,cooo,Pt,preserve_rng_state=False) #+loss_grad+(torch.sum(deformL2a**2)+torch.sum(deformR2a**2))*1e-9+(torch.sum(deformL4a**2)+torch.sum(deformR4a**2))*1e-9+(torch.sum(deformL8a**2)+torch.sum(deformR8a**2))*1e-9

                    loss_L=0
                    loss_R=0
                    loss_L0=0
                    loss_R0=0


                    loss=(loss_for)*2+loss_grad0*30
                    loss.backward()

                optimizer0.step(closure_adj)
                optimizer1.step(closure_adj)
                import imageio
                imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=10)
                imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=10)
                imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(50)], fps=10)
        return self.deformL_param_adj,self.deformR_param_adj

    def for_field_solver(self,inner_iter,T1,T2):
     from utils_reg1 import flows,_updateb,f
     from torch.utils import checkpoint
     import torch
     import numpy as np
     import random
     for i in range(3):
        self.deformL_param_adj[i]=self.deformL_param_adj[i].detach().cpu().numpy()
        self.deformR_param_adj[i]=self.deformR_param_adj[i].detach().cpu().numpy()
        self.deformL_param_adj[i]=torch.from_numpy(self.deformL_param_adj[i]).cuda()
        self.deformR_param_adj[i]=torch.from_numpy(self.deformR_param_adj[i]).cuda()
        self.deformL_param_for[i]=torch.nn.parameter.Parameter(self.deformL_param_for[i],requires_grad=True)
        self.deformR_param_for[i]=torch.nn.parameter.Parameter(self.deformR_param_for[i],requires_grad=True)
     optimizer2=torch.optim.LBFGS([self.deformL_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 

     optimizer3=torch.optim.LBFGS([self.deformR_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 
     deform_look=np.zeros([T2-T1,self.mps.shape[2],self.mps.shape[3]])
     image_still=np.zeros([T2-T1,self.mps.shape[2],self.mps.shape[3]])
     image_look=np.zeros([T2-T1,self.mps.shape[2],self.mps.shape[3]])
     P=torch.ones([40,1])
     mps=torch.from_numpy(self.mps).cpu()
     def closure_for():
        from utils_reg1 import flows,_updateb,f
        from torch.utils import checkpoint
        import torch
        import numpy as np
        import random
        with torch.no_grad():
            loss_for=0
            loss_grad0=0
            loss_det=0
            loss_rev1=0

            for t in K:
                print(t)
           # optimizer0.zero_grad()
           # optimizer1.zero_grad()
                #flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
                count=0
                lossz=np.zeros([20])

                #loss_for=0
               # loss_rev=0
               # print(j)
             #   print(j)
               # im_rem=torch.zeros([10,mps.shape[1],mps.shape[2],mps.shape[3]],dtype=torch.cfloat)
                #deformb=flows(deformL_param_adj,deformR_param_adj,j)
                #deforma=torch.nn.functional.interpolate(deformb, size=[mps.shape[1],mps.shape[2],mps.shape[3]], 
                #_factor=None, mode='trilinear', align_corners=False, recompute_scale_factor=None)*3
                flowa=self.get_adj_field(t)
                flowa_inv=self.get_for_field(t)
                im_test = 0
                for j in range(self.J):
                        im_test += self.B[j](self.L[j] * self.R[j][t])
                im_test=torch.as_tensor(im_test,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                im_test=im_test
                im_test=torch.reshape(im_test,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
                
             
             
                vectors=[]
                shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
                size=shape
                vectors = [ torch.arange(0, s) for s in size ] 
                #vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
               # shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locs = new_locs[..., [2,1,0]]
                ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
              
               
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [2,1,0]]
                im_inv_real=torch.nn.functional.grid_sample(torch.real(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv_imag=torch.nn.functional.grid_sample(torch.imag(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
              #  ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='zeros', align_corners=True)

           # im_out=torch.complex(torch.utils.checkpoint.checkpoint(warp,torch.real(im_test), mgrid0.cuda(),preserve_rng_state=False),torch.utils.checkpoint.checkpoint(warp,torch.imag(im_test), mgrid0.cuda(),preserve_rng_state=False)).cuda()
            #im_rev=torch.complex(torch.utils.checkpoint.checkpoint(warp,torch.real(im_out), -flowa.cuda(),preserve_rng_state=False),torch.utils.checkpoint.checkpoint(warp,torch.imag(im_out), -flowa.cuda(),preserve_rng_state=False)).cuda()
           # diff=im_rev-im_test
           # loss_rev=torch.norm(diff,2)
           # flow_inv=torch.utils.checkpoint.checkpoint(inverse_field,flowa.cuda())
           # id_guess=warp1(flowa.cuda(),flow_inv.cuda())
              #  loss_self1=torch.nn.MSELoss()
             # 3  loss_rev1=loss_self1(torch.squeeze(im_inv),torch.squeeze(torch.abs(im_test)))
                   # im_test=im_out.detach().cpu().numpy()
                   # im_out=torch.complex(torch.utils.checkpoint.checkpoint(warp,torch.real(im_test), flowa.cuda(),preserve_rng_state=False),torch.utils.checkpoint.checkpoint(warp,torch.imag(im_test), flowa.cuda(),preserve_rng_state=False)).cuda()
               # im_rev=torch.complex(torch.utils.checkpoint.checkpoint(warp,torch.real(im_out), -flowa.cuda(),preserve_rng_state=False),torch.utils.checkpoint.checkpoint(warp,torch.imag(im_out), -flowa.cuda(),preserve_rng_state=False)).cuda()
               # diff=im_rev-im_test
               # loss_rev=torch.norm(diff,2)
               # id_guess=warp(flowa.cuda(),-flowa.cuda())
               ## loss_self1=torch.nn.MSELoss()
                #loss_rev1=loss_self1(id_guess,torch.zeros_like(id_guess))
               # im_rev=torch.complex(torch.utils.checkpoint.checkpoint(warp,torch.real(im_out), flowa_rev[j:j+1].cuda(),preserve_rng_state=False),torch.utils.checkpoint.checkpoint(warp,torch.imag(im_out), flowa_rev[j:j+1].cuda(),preserve_rng_state=False)).cuda()
               # diff=im_rev-im_test
               # loss_rev=torch.norm(diff,2)
               # id_guess=warp(flowa.cuda(),deforma_rev[j:j+1].cuda())
                im_test_ref=0
                for j in range(self.J):
                    im_test_ref += self.B[j](self.L[j] * self.R[j][0])
                im_test_ref=torch.as_tensor(im_test_ref,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test_ref).max()
                im_test_ref=im_test_ref
                diff=torch.squeeze(im_inv)-torch.squeeze(im_test_ref)
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
               # loss_rev1=loss_self1(id_guess,torch.zeros_like(id_guess))
                
                loss_grad0=loss_grad0+torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())
              
                cooo=torch.ones([1])*lo
              #  loss_for=loss_for+torch.utils.checkpoint.checkpoint(_updateb,im_out,ksp_ta,dcf_t,coord_t,mps,cooo,Pt,preserve_rng_state=False) #+loss_grad+(torch.sum(deformL2a**2)+torch.sum(deformR2a**2))*1e-9+(torch.sum(deformL4a**2)+torch.sum(deformR4a**2))*1e-9+(torch.sum(deformL8a**2)+torch.sum(deformR8a**2))*1e-9
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
               
            loss=loss_rev1*1e-1 #+loss_grad0*30 #+loss_R*1e-6 #loss_L0*1e-8+loss_R0*1e-8 
            return loss

     for io in range(inner_iter):

            
            optimizer2.zero_grad()
            optimizer3.zero_grad()
       
            loss_grad0=0
            loss_tot=0
            loss_for=0
            loss_rev=0
            loss_for=0
            loss_grad0=0
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])
          
            K=random.sample(range(T1,T2), T2-T1)
            for t in K:
                print(t)
                loss_rev1=0

                #print(j)
               # optimizer0.zero_grad()
                flowa=self.get_adj_field(t)
                flowa_inv=self.get_for_field(t)
                im_test = 0
                for j in range(self.J):
                        im_test += self.B[j](self.L[j] * self.R[j][t])
                im_test=torch.as_tensor(im_test,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                im_test=im_test
                im_test=torch.reshape(im_test,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])

               
                shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locs = new_locs[..., [2,1,0]]
                ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
               
                #ima=torch.nn.functional.grid_sample(im_test, new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
              

                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                for i in range(len(shape)):
                    new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [2,1,0]]
                im_inv_real=torch.nn.functional.grid_sample(torch.real(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv_imag=torch.nn.functional.grid_sample(torch.imag(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
                #im_inv=torch.nn.functional.grid_sample(im_out, new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_test_ref=0
                for j in range(self.J):
                    im_test_ref += self.B[j](self.L[j] * self.R[j][0])
                im_test_ref=torch.as_tensor(im_test_ref,device='cuda').unsqueeze(0) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test_ref).max()
                im_test_ref=im_test_ref
                diff=torch.squeeze(im_inv)-torch.squeeze(im_test_ref)
                loss_rev1=torch.norm(diff,2)**2
               # loss_self1=torch.nn.MSELoss()
                #loss_rev1=loss_self1(torch.squeeze(im_inv),torch.squeeze(torch.abs(im_test)))*10


                if t>=T1 and t<T2:
                    deform_look[t-T1]=np.abs(flowa_inv[:,0,:,:,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                    image_still[t-T1]=np.abs(im_inv[0,0,:,110,:].detach().cpu().numpy())
                    image_look[t-T1]=np.abs(im_out[0,0,:,110,:].detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_test.detach().cpu().numpy())

                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())

                lo=0

                loss=loss_rev1*1e-1 #+loss_grad0*30
                print(loss)

                loss.backward()



            optimizer2.step(closure_for)
            optimizer3.step(closure_for)

            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=10)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=10)
            imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(50)], fps=10)
     return self.deformL_param_for,self.deformR_param_for


    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                lossa=0
                if self.epoch<=80:
                    for i, t in enumerate(np.random.permutation(self.T)):
                        loss += self._update(t)

                       # lossa+=self._updatea(t)


                      #  print(lossa)
                        pbar.set_postfix(loss=(loss) * self.T / (i + 1))
                        pbar.update()
                    #self.optimizer.step()
                    #self.optimizer.zero_grad()
                    import imageio
                    import imageio
                    
                    imageio.mimsave('image.gif', [np.abs(self.deform_look[i,:,:])*1e15 for i in range(50)], fps=10)
                if self.epoch>80:
                    self.deformL_param_for,self.deformR_param_for=self.for_field_solver(3,0,50)
                    for i in range(1):
                        for i, t in enumerate(np.random.permutation(self.T)):
                            loss += self._update(t)

                           # lossa+=self._updatea(t)


                          #  print(lossa)
                            pbar.set_postfix(loss=(loss) * self.T / (i + 1))
                            pbar.update()
                        #self.optimizer.step()
                        #self.optimizer.zero_grad()
                        import imageio
                        import imageio
                        imageio.mimsave('image.gif', [np.abs(self.deform_look[i,:,50,:])*1e15 for i in range(50)], fps=5)
                    self.deformL_param_adj,self.deformR_param_adj=self.adj_field_solver(3,200,0,50)
               
               
                
               
           

    def _update(self, t):
        # Form image.
        mps=self.mps
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])
        if t>=0 and t<50:
            self.deform_look[t]=np.abs(img_t[:,110,:].get())
        flow_adj=self.get_adj_field(t)
        img_t=torch.as_tensor(img_t,device='cuda')
        img_t=self.warp(flow_adj,img_t).cuda()
        print('a')
        img_t = xp.asarray(img_t.detach().cpu().numpy())
        print('b')

# Convert it into a CuPy array.
       # img_t = cupy.from_dlpack(img_t)
        #img_t=cupy.asarray(img_t)

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc
       
        e_t=torch.as_tensor(e_t,device='cuda') 
        flow_for=self.get_for_field(t)
        e_t=self.warp(flow_for,e_t)
        e_t = cupy.asarray(e_t.detach().cpu().numpy())
       # e_t = cupy.from_dlpack(e_t)
       # e_t=cupy.asarray(e_t)
        #e_tcorrected=Minv(e_t)

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # L gradient.
            #Minv*A^H*
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Precondition.
            g_L_j /= self.J * self.sigma[j] + lamda_j
            g_R_jt /= self.J * self.sigma[j] + lamda_j

            # Add.
            self.L[j] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t
    '''  
    def _updatea(self,t):
    # Form image.
        img_t = 0
        f=Grad(penalty='l2')
        img_t = 0
        f=Grad(penalty='l2')
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])
        img_t=torch.from_numpy(img_t.get())
        img_t=img_t.cuda()

       # img_t=torch.from_numpy(ima)
       # img_t=img_t.cuda()
        img_t=torch.reshape(img_t,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        print(self.deform_adjoint.shape)
        img_treal=warp(torch.reshape(torch.real(img_t),[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]]),self.deform_adjoint[t:t+1].cuda())
        img_timag=warp(torch.reshape(torch.imag(img_t),[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]]),self.deform_adjoint[t:t+1].cuda())  
        #deform_adjointa=tf.transpose(deform_adjoint1.detach().cpu().numpy(),[0,2,3,4,1])
       # with tf.device('gpu:0'):
       #     jac=compute_jacobian_determinant(deform_adjointa)
       # jac1=torch.zeros([1,mps.shape[1],mps.shape[2],mps.shape[3]])
       # jac1[:,:mps.shape[1]-2,:mps.shape[2]-2,:mps.shape[3]-2]=torch.squeeze(torch.from_numpy(jac.numpy()))
       # jac1=torch.reshape(jac1,[1,1,mps.shape[1],mps.shape[2],mps.shape[3]])
        img_treal=img_treal.cuda() #*jac1.cuda()
        img_timag=img_timag.cuda() #*jac1.cuda()
        img_ta=torch.complex(img_treal,img_timag)
        warped=torch.squeeze(img_ta)
        #img_t=cupy.asarray(img_t.detach().cpu().numpy())

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)
        coord_t=torch.from_numpy(coord_t.get())
        dcf_t=torch.from_numpy(dcf_t.get())
        ksp_t=torch.from_numpy(ksp_t.get())

        # Data consistency.
        loss_t=0
        for c in range(self.C):

            loss_t=loss_t+calculate_sense0(warped,ksp_t[c],self.mps[c],coord_t,dcf_t)
        '''


       # import imageio
       # imageio.mimsave('./powers21.gif', [np.abs(np.array(self.deform[i,0,:,:,110].detach().cpu().numpy()))*1e15 for i in range(10)], fps=8)
      #  imageio.mimsave('./powers22.gif', [np.abs(np.array(self.deform_adjoint[i,0,:,:,110].detach().cpu().numpy()))*1e15 for i in range(10)], fps=8)
       # loss_t = loss_t.item()
      
   
        
            
            
            
            
            
            

    
            
            
            
            
            
            
           
    

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf):
        import sigpy
        import torch
        #ksp=torch.from_numpy(ksp)
        ksp=torch.reshape(ksp.cpu(),[-1])
        coord_t=torch.reshape(coord_t,[-1,3])
        dcf=torch.reshape(dcf,[-1])
       
       
      
       # coord_t=torch.from_numpy(coord_t)
        coord_t=coord_t.cpu()
        #3,mps_c,coord_t,dcf
        dcf=dcf.cpu()
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device1 = torch.device("cpu")
        F = sigpy.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t, oversamp=1.25, width=2)
        F_torch = sigpy.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sigpy.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)


        dcf_t = torch.zeros(dcf.shape[0], 2).cpu()
        dcf_t[:, 0] = torch.squeeze((dcf)).cpu()
        dcf_t[:,1]=torch.squeeze((dcf).cpu())
        ksp_tca = torch.zeros(ksp.shape[0], 2).cpu()
       
        ksp_tca[:, 0] = torch.real((ksp)).cpu()
        ksp_tca[:, 1] = torch.imag((ksp)).cpu()
        mps_c1 = torch.zeros(mps_c.shape[0], mps_c.shape[1], mps_c.shape[2], 2, dtype=torch.float).cpu()
        mps_c1[:, :, :, 0] = torch.squeeze(torch.real(mps_c)).cpu()
        mps_c1[:, :, :, 1] = torch.squeeze(torch.imag(mps_c)).cpu()
        
        diff=torch.zeros([mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],2]).cpu()
        diff[:,:,:,0]=torch.real(M_t.cpu()* mps_c.to(device1) )
        diff[:,:,:,1]=torch.imag(M_t.cpu()* mps_c.to(device1) )
        e_tctot=torch.zeros([mps_c.shape[0],mps_c.shape[1],mps_c.shape[2]],dtype=torch.cfloat).cpu()



        c1 = torch.zeros([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2], 2], device=device1)
        c = diff.to(device1) 
        c1[:, :, :, 0] = c[:, :, :, 0].cpu()
        c1[:, :, :, 1] = c[:, :, :, 1].cpu()
        c1 = c1.cuda()


      
        e_tc=F_torch.apply(c1.cuda())
        ksp_t=torch.complex(ksp_tca[:,0],ksp_tca[:,1])
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
         
        #loss_self1=torch.nn.MSELoss()
        resk=(ksp_tca.cuda()-e_tc.cuda())*dcf_t.cuda()**0.5
        loss=torch.norm(resk)**2 #torch.abs(torch.sum((ksp_t.cuda()*dcf.cuda()**0.5-e_tca.cuda()*dcf.cuda()**0.5)**2))
       
        
        
       
        return loss
def _repeat(x, n_repeats):
    rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
    return tf.reshape(rep, [-1])

def _interpolate13d(imgs, x, y, z):
    batch_size=tf.shape(imgs)[0]
   
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[4]

    x_t, y_t, z_t = tf.meshgrid(tf.linspace(-1., 1.,xlen),
                                tf.linspace(-1., 1.,ylen),
                                tf.linspace(-1., 1., zlen), indexing='ij')
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    z_t_flat = tf.reshape(z_t, (1, -1))
    x_t_flat = tf.tile(x_t_flat, [batch_size, 1])
    y_t_flat = tf.tile(y_t_flat, [batch_size, 1])
    z_t_flat = tf.tile(z_t_flat, [batch_size, 1])
    x_t_flat = tf.reshape(x_t_flat, [xlen*ylen*zlen*batch_size])
    y_t_flat = tf.reshape(y_t_flat, [xlen*ylen*zlen*batch_size])
    z_t_flat = tf.reshape(z_t_flat, [xlen*ylen*zlen*batch_size])
    x = x_t_flat + x
    y = y_t_flat + y
    z = z_t_flat + z

    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    z = tf.cast(z,tf.float32)
    xlen_f = tf.cast(xlen,tf.float32)
    ylen_f = tf.cast(ylen,tf.float32)
    zlen_f = tf.cast(zlen,tf.float32)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    max_z = tf.cast(zlen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)
    base = _repeat(tf.range(n_batch) * xlen * ylen * zlen,
                   xlen * ylen * zlen)
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.cast(imgs_flat,tf.float32)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.cast(x0,tf.float32)
    dy = y - tf.cast(y0,tf.float32)
    dz = z - tf.cast(z0,tf.float32)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    output = tf.reshape(output, [n_batch, xlen, ylen, zlen, n_channel])

    return output


def bilinear_interp3d(vol, x, y, z, out_size, edge_size=0):
    batch_size, depth, height, width, channels = vol.get_shape().as_list()

    if edge_size > 0:
        vol = tf.pad(vol, [[0, 0], [edge_size, edge_size], [edge_size, edge_size], [edge_size, edge_size], [0, 0]],
                     mode='CONSTANT')
    z_t, y_t, x_t = tf.meshgrid(tf.linspace(-1., 1.,66 ),
                                tf.linspace(-1., 1., 65),
                                tf.linspace(-1., 1., 65), indexing='ij')
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    z_t_flat = tf.reshape(z_t, (1, -1))
    x_t_flat = tf.tile(x_t_flat, [batch_size, 1])
    y_t_flat = tf.tile(y_t_flat, [batch_size, 1])
    z_t_flat = tf.tile(z_t_flat, [batch_size, 1])
    x_t_flat = tf.reshape(x_t_flat, [66*65*65* batch_size])
    y_t_flat = tf.reshape(y_t_flat, [66*65*65 * batch_size])
    z_t_flat = tf.reshape(z_t_flat, [66*65*65 * batch_size])
    x = x_t_flat + x
    y = y_t_flat + y
    z = z_t_flat + z

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)

    depth_f = tf.cast(depth, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)

    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]

    # scale indices to [0, width/height/depth - 1]
    x = (x + 1.) / 2. * (width_f - 1.)
    y = (y + 1.) / 2. * (height_f - 1.)
    z = (z + 1.) / 2. * (depth_f - 1.)

    # clip to to [0, width/height/depth - 1] +- edge_size
    x = tf.clip_by_value(x, -edge_size, width_f - 1. + edge_size)
    y = tf.clip_by_value(y, -edge_size, height_f - 1. + edge_size)
    z = tf.clip_by_value(z, -edge_size, depth_f - 1. + edge_size)

    x += edge_size
    y += edge_size
    z += edge_size

    # do sampling
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    z0_f = tf.floor(z)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    z1_f = z0_f + 1

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    z0 = tf.cast(z0_f, tf.int32)

    x1 = tf.cast(tf.minimum(x1_f, width_f - 1. + 2 * edge_size), tf.int32)
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1. + 2 * edge_size), tf.int32)
    z1 = tf.cast(tf.minimum(z1_f, depth_f - 1. + 2 * edge_size), tf.int32)

    dim3 = (width + 2 * edge_size)
    dim2 = (width + 2 * edge_size) * (height + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size) * (depth + 2 * edge_size)

    base = _repeat(tf.range(batch_size) * dim1, out_depth * out_height * out_width)
    base_z0 = base + z0 * dim2
    base_z1 = base + z1 * dim2

    base_y00 = base_z0 + y0 * dim3
    base_y01 = base_z0 + y1 * dim3
    base_y10 = base_z1 + y0 * dim3
    base_y11 = base_z1 + y1 * dim3

    idx_000 = base_y00 + x0
    idx_001 = base_y00 + x1
    idx_010 = base_y01 + x0
    idx_011 = base_y01 + x1
    idx_100 = base_y10 + x0
    idx_101 = base_y10 + x1
    idx_110 = base_y11 + x0
    idx_111 = base_y11 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    vol_flat = tf.reshape(vol, [-1, channels])
    I000 = tf.gather(vol_flat, idx_000)
    I001 = tf.gather(vol_flat, idx_001)
    I010 = tf.gather(vol_flat, idx_010)
    I011 = tf.gather(vol_flat, idx_011)
    I100 = tf.gather(vol_flat, idx_100)
    I101 = tf.gather(vol_flat, idx_101)
    I110 = tf.gather(vol_flat, idx_110)
    I111 = tf.gather(vol_flat, idx_111)

    # and finally calculate interpolated values
    w000 = tf.expand_dims((z1_f - z) * (y1_f - y) * (x1_f - x), 1)
    w001 = tf.expand_dims((z1_f - z) * (y1_f - y) * (x - x0_f), 1)
    w010 = tf.expand_dims((z1_f - z) * (y - y0_f) * (x1_f - x), 1)
    w011 = tf.expand_dims((z1_f - z) * (y - y0_f) * (x - x0_f), 1)
    w100 = tf.expand_dims((z - z0_f) * (y1_f - y) * (x1_f - x), 1)
    w101 = tf.expand_dims((z - z0_f) * (y1_f - y) * (x - x0_f), 1)
    w110 = tf.expand_dims((z - z0_f) * (y - y0_f) * (x1_f - x), 1)
    w111 = tf.expand_dims((z - z0_f) * (y - y0_f) * (x - x0_f), 1)

    output = tf.add_n([
        w000 * I000,
        w001 * I001,
        w010 * I010,
        w011 * I011,
        w100 * I100,
        w101 * I101,
        w110 * I110,
        w111 * I111])
    return output
