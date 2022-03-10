import sigpy
from math import ceil

def Boperator(blk_widths,mps):
        D=3
        B=[]
        device=sp.Device(0)
        for j in range(2):
            img_shape=[mps.shape[1],mps.shape[2],mps.shape[3]]
            b_j = [min(i, blk_widths[j]) for i in img_shape]
            s_j = [(b + 1) // 2 for b in b_j]

            i_j = [ceil((i - b + s) / s) * s + b - s
                   for i, b, s in zip(img_shape, b_j, s_j)]

            C_j = sp.linop.Resize(img_shape, i_j,
                                  ishift=[0] * D, oshift=[0] * D)
            B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
            with device:
                w_j = sp.hanning(b_j, dtype=cupy.complex64, device=device)**0.5
            W_j = sp.linop.Multiply(B_j.ishape, w_j)
            B.append(C_j * B_j * W_j)
        return B
        