import torch , triton
import triton.language as tl

#gelu
@triton.jit
def Gelukernel(xptr , outptr , M , K , xstrm , xstrn , outstrx , outstry , MBLOCK: tl.constexpr , KBLOCK: tl.constexpr):
    pid=tl.program_id(axis=0)

    row=pid*MBLOCK+tl.arange(0,MBLOCK)
    offsets=tl.arange(0,KBLOCK)

    ptrx=xptr+row[:,None]*xstrm+offsets[None,:]*xstrn
    out=outptr+row[:,None]*outstrx+offsets[None,:]*outstry
    sqrt_2_pi=0.7978845608028654

    for k in range(0,K,KBLOCK):
        maskx=(row[:,None]<M)&((k+offsets)[None,:]<K)

        x=tl.load(ptrx,mask=maskx,other=0.0)
        inner=sqrt_2_pi*(x+0.044715*x*x*x)

        res = 0.5*x*(1.0+(2.0*tl.sigmoid(2.0*inner)-1.0))
        tl.store(out,res,mask=maskx)

        ptrx+=KBLOCK*xstrn
        out+=KBLOCK*outstry

def gelu(x):

    original_shape = x.shape
    
    if len(original_shape) == 1:
        x_2d = x.view(1, -1)
    else:
        x_2d = x.view(-1, original_shape[-1])
    
    M, K = x_2d.shape

    BLOCK_M = 128
    BLOCK_K = 32
 
    out_2d = torch.empty_like(x_2d)
    
    grid = (triton.cdiv(M, BLOCK_M),)
    Gelukernel[grid](
        x_2d, out_2d,
        M, K,
        x_2d.stride(0), x_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_M, BLOCK_K
    )

    return out_2d.view(original_shape)

def fusedresidukernel(xptr , resptr , outptr , eps , gamma , beta , M , K , xstrm , xstrn , resstrm , resstrn , outstrm , outstrn , gammastry , betastry , MBLOCK:tl.constexpr , KBLOCK:tl.constexpr):
    pid = tl.program_id(axis = 0)

    row = pid * MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)

    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptrres = resptr + row[: , None] * resstrm + offsets[None , :] * resstrn

    mean = tl.zeros((MBLOCK,) , dtype = tl.float32)
    M2 = tl.zeros((MBLOCK,) , dtype = tl.float32) 

    count = tl.zeros((MBLOCK,) , dtype = tl.float32) 

    for k in range(0 , K , KBLOCK):
        mask = (row[: , None] < M) & ((k + offsets[None , :]) < K)

        x = tl.load(ptrx , mask = mask , other = 0.0)
        res = tl.load(ptrres , mask = mask , other = 0.0)

        valid = mask.to(tl.float32)

        final = x + res # i think we shouldn't use tl.sum as its for a single tensor and we need element wise addition
        count += tl.sum(valid , axis = 1)

        diff = (final - mean[: , None]) * valid
        delta = tl.sum(diff , axis = 1)

        mean += delta / count
        M2 = M2 + tl.sum(diff * (final - mean[: , None]) * valid , axis = 1)

        ptrx += KBLOCK * xstrn
        ptrres += KBLOCK * resstrn

    var = M2 / K
    st = tl.rsqrt(var + eps)
    
    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptrres = resptr + row[: , None] * resstrm + offsets[None , :] * resstrn
    gammaptr = gamma + offsets[None, :] * gammastry
    betaptr  = beta  + offsets[None, :] * betastry

    out = outptr + row[: , None] * outstrm + offsets[None , :] * outstrn


    for k in range(0 , K , KBLOCK):
        mask = (row[: , None] < M) & ((k + offsets[None , :]) < K)

        x = tl.load(ptrx , mask = mask , other = 0.0)
        res = tl.load(ptrres , mask = mask , other = 0.0)

        final = x + res 

        maskg = (k + offsets)[None, :] < K       
        gama  = tl.load(gammaptr, mask=maskg, other=0.0)
        bet   = tl.load(betaptr,  mask=maskg, other=0.0)

        norm = ((final - mean[: , None]) * st[: , None]) * gama + bet
        tl.store(out , norm , mask = mask)

        out += KBLOCK * outstrn
        ptrx += KBLOCK * xstrn
        ptrres += KBLOCK * resstrn
        gammaptr += KBLOCK * gammastry
        betaptr  += KBLOCK * betastry


def layernormwithresidue(x , res , gamma , beta , eps):
    original_shape = x.shape
    res_original = res.shape
    
    if len(original_shape) == 1:
        x_2d = x.view(1, -1)
        res_2d = res.view(1 , -1)
    else:
        x_2d = x.view(-1, original_shape[-1])
        res_2d = res.view(-1, res_original[-1])

    M , K = x_2d.shape
    out = torch.empty_like(x_2d)

    mblock = 128
    kblock = 32

    eps = eps if eps else 1e-5

    grid = ((M + mblock - 1) // mblock,)

    fusedresidukernel[grid](x_2d , res_2d , out , eps , gamma , beta , M , K , x_2d.stride()[0] , x_2d.stride()[1] , res_2d.stride()[0] , res_2d.stride()[1] , out.stride()[0] , out.stride()[1] , gamma.stride()[0] , beta.stride()[0] , mblock , kblock)
    return out.view(original_shape)