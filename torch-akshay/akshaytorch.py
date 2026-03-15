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


@triton.jit
def matmulkernel(xptr , yptr , outptr , M , N , K , mstrm , mstrk , nstrk , nstrn , kstrm , kstrn , blockm : tl.constexpr , blockn : tl.constexpr , blockk : tl.constexpr):
    xpid = tl.program_id(axis = 0)
    ypid = tl.program_id(axis = 1)

    row = xpid * blockm + tl.arange(0 , blockm)
    col = ypid * blockn + tl.arange(0 , blockn)
    rk = tl.arange(0 , blockk)

    ptr_x = xptr + (row[: , None] * mstrm) + (rk[None , :] * mstrk)
    ptr_y = yptr + (rk[: , None] * nstrk) + (col[None , :] * nstrn)

    acc = tl.zeros((blockm , blockn) , dtype = tl.float32)

    for k in range(0 , K , blockk):
        maskx = (row[: , None] < M) & ((k + rk[None , :]) < K)
        x = tl.load(ptr_x , mask = maskx , other = 0.0)

        masky = ((k + rk[: , None]) < K) & (col[None , :] < N)
        y = tl.load(ptr_y , mask = masky , other = 0.0)

        acc += tl.dot(x , y)

        ptr_x += blockk * mstrk
        ptr_y += blockk * nstrk
    
    out = outptr + (row[: , None] * kstrm) + (col[None , :] * kstrn)
    mask = (row[: , None] < M) & (col[None , :] < N)
    tl.store(out , acc.to(tl.float32) , mask = mask)

def matmul(x, y):
    orig_x_shape = x.shape
    orig_y_shape = y.shape

    x_nd = x.unsqueeze(0) if len(orig_x_shape) == 1 else x
    y_nd = y.unsqueeze(-1) if len(orig_y_shape) == 1 else y

    x_batch = x_nd.shape[:-2] if len(x_nd.shape) > 2 else torch.Size([])
    y_batch = y_nd.shape[:-2] if len(y_nd.shape) > 2 else torch.Size([])
    batch   = torch.broadcast_shapes(x_batch, y_batch)

    x_b = x_nd.expand(*batch, *x_nd.shape[-2:]).reshape(-1, x_nd.shape[-2], x_nd.shape[-1])
    y_b = y_nd.expand(*batch, *y_nd.shape[-2:]).reshape(-1, y_nd.shape[-2], y_nd.shape[-1])

    B  = x_b.shape[0]
    M  = x_b.shape[1]
    K  = x_b.shape[2]
    N  = y_b.shape[2]
    assert K == y_b.shape[1], f"Incompatible: x has K={K} but y has K={y_b.shape[1]}"

    mblock, nblock, kblock = 128, 128, 32
    out = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    for b in range(B):
        xb   = x_b[b]   # (M, K)
        yb   = y_b[b]   # (K, N)
        outb = out[b]    # (M, N)

        grid = (triton.cdiv(M, mblock), triton.cdiv(N, nblock))

        matmulkernel[grid](
            xb, yb, outb,
            M, N, K,
            xb.stride(0),   xb.stride(1),
            yb.stride(0),   yb.stride(1),
            outb.stride(0), outb.stride(1),
            mblock, nblock, kblock
        )

    if len(orig_x_shape) == 1 and len(orig_y_shape) == 1:
        return out.view(())
    elif len(orig_x_shape) == 1 and len(orig_y_shape) == 2:
        return out.view(N)
    elif len(orig_x_shape) == 2 and len(orig_y_shape) == 1:
        return out.view(M)
    else:
        return out.view(*batch, M, N)


#relu 
@triton.jit
def ReLUkernel(xptr , outptr , M , K , xstrm , xstrn , outstrx , outstry , MBLOCK: tl.constexpr , KBLOCK: tl.constexpr):
    pid = tl.program_id(axis = 0)
    row = pid  *MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)

    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptrout = outptr + row[: , None] * outstrx + offsets[None , :] * outstry

    for k in range(0 , K , KBLOCK):
        mask = (row[: , None] < M) & ((k + offsets)[None , :] < K)

        data = tl.load(ptrx , mask = mask , other = 0.0)
        x = tl.where(data > 0 , data , 0.0)

        tl.store(ptrout , x , mask = mask)
        ptrx += KBLOCK * xstrn
        ptrout += KBLOCK * outstry

def ReLU(x):
    org_shape = x.shape
    if len(org_shape) == 1:
        x_2d = x.view(1 , -1)
    if len(org_shape) == 2:
        x_2d = x
    if len(org_shape) > 2:
        x_2d = x.view(-1 , x.shape[-1])

    M , K = x_2d.shape
    Mblock = 128
    kblock = 32

    out = torch.empty_like(x_2d)

    grid = ((M + Mblock - 1) // Mblock ,)

    ReLUkernel[grid](x_2d , out , M , K , x_2d.stride()[0] , x_2d.stride()[1] , out.stride()[0] , out.stride()[1] , Mblock , kblock)

    return out.view(org_shape)


# concat
@triton.jit
def concatkernel(xptr , yptr , outptr , M , K , xstrm , xstrn , ystrm , ystrn , outstrx , outstry , MBLOCK: tl.constexpr , KBLOCK: tl.constexpr):
    pid = tl.program_id(axis = 0)
    row = pid * MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)
    catoffsets = tl.arange(0 , KBLOCK)

    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptry = yptr + row[: , None] * ystrm + offsets[None , :] * ystrn

    ptroutleft = outptr + row[: , None] * outstrx + catoffsets[None , :] * outstry
    ptroutright = ptroutleft + K * outstry

    for k in range(0 , K , KBLOCK):
        maskuniversal = (row[: , None] < M) & ((k + offsets)[None , :] < K)

        left = tl.load(ptrx , mask = maskuniversal , other = 0.0)
        right = tl.load(ptry , mask = maskuniversal , other = 0.0)


        maskout = (row[:, None] < M) & ((k + offsets)[None, :] < K)

        tl.store(ptroutleft , left , mask = maskout)
        tl.store(ptroutright , right , mask = maskout)
        ptrx += KBLOCK * xstrn
        ptry += KBLOCK * ystrn
        ptroutleft += KBLOCK * outstry
        ptroutright += KBLOCK * outstry

def cat(x , y):
    org_shape = x.shape
    if len(org_shape) == 1:
        x_2d = x.view(1 , -1)
    if len(org_shape) == 2:
        x_2d = x
    if len(org_shape) > 2:
        x_2d = x.view(-1 , x.shape[-1])

    org_y = y.shape
    if len(org_y) == 1:
        y_2d = y.view(1 , -1)
    if len(org_y) == 2:
        y_2d = y
    if len(org_y) > 2:
        y_2d = y.view(-1 , y.shape[-1])

    M , K = x_2d.shape
    out = torch.empty((M , 2*K) , device = 'cuda')

    mblock , kblock = 128 , 32

    grid = ((M + mblock - 1) // mblock,)

    concatkernel[grid](x , y , out , M , K , x_2d.stride()[0] , x_2d.stride()[1] , y_2d.stride()[0] , y_2d.stride()[1] , out.stride()[0] , out.stride()[1] , mblock , kblock)
    if len(org_shape) == 1:
        return out.view(-1)

    if len(org_shape) == 2:
        return out

    if len(org_shape) > 2:
        new_shape = list(org_shape)
        new_shape[-1] *= 2
        return out.view(*new_shape)
    

# concat
@triton.jit
def concatkernel(xptr , yptr , outptr , M , K , xstrm , xstrn , ystrm , ystrn , outstrx , outstry , MBLOCK: tl.constexpr , KBLOCK: tl.constexpr):
    pid = tl.program_id(axis = 0)
    row = pid * MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)
    catoffsets = tl.arange(0 , KBLOCK)

    ptrx = xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptry = yptr + row[: , None] * ystrm + offsets[None , :] * ystrn

    ptroutleft = outptr + row[: , None] * outstrx + catoffsets[None , :] * outstry
    ptroutright = ptroutleft + K * outstry

    for k in range(0 , K , KBLOCK):
        maskuniversal = (row[: , None] < M) & ((k + offsets)[None , :] < K)

        x = tl.load(ptrx , mask = maskuniversal , other = 0.0)
        y = tl.load(ptry , mask = maskuniversal , other = 0.0)

        left = tl.where(x > 0 , x , 0.0)
        right = tl.where(y > 0 , y , 0.0)

        maskout = (row[:, None] < M) & ((k + offsets)[None, :] < K)

        tl.store(ptroutleft , left , mask = maskout)
        tl.store(ptroutright , right , mask = maskout)
        ptrx += KBLOCK * xstrn
        ptry += KBLOCK * ystrn
        ptroutleft += KBLOCK * outstry
        ptroutright += KBLOCK * outstry

def fusedcat(x , y):
    org_shape = x.shape
    if len(org_shape) == 1:
        x_2d = x.view(1 , -1)
    if len(org_shape) == 2:
        x_2d = x
    if len(org_shape) > 2:
        x_2d = x.view(-1 , x.shape[-1])

    org_y = y.shape
    if len(org_y) == 1:
        y_2d = y.view(1 , -1)
    if len(org_y) == 2:
        y_2d = y
    if len(org_y) > 2:
        y_2d = y.view(-1 , y.shape[-1])

    M , K = x_2d.shape
    out = torch.empty((M , 2*K) , device = 'cuda')

    mblock , kblock = 128 , 32

    grid = ((M + mblock - 1) // mblock,)

    concatkernel[grid](x , y , out , M , K , x_2d.stride()[0] , x_2d.stride()[1] , y_2d.stride()[0] , y_2d.stride()[1] , out.stride()[0] , out.stride()[1] , mblock , kblock)
    if len(org_shape) == 1:
        return out.view(-1)

    if len(org_shape) == 2:
        return out

    if len(org_shape) > 2:
        new_shape = list(org_shape)
        new_shape[-1] *= 2
        return out.view(*new_shape)
    

import torch , triton
import triton.language as tl

@triton.jit 
def groupnormkernel(xptr , outptr , eps , M , K , xstrm , xstrn , outstrm , outstrn , batchstr , groupstr , MBLOCK: tl.constexpr , KBLOCK: tl.constexpr):
    batchpid = tl.program_id(axis = 0)
    grouppid = tl.program_id(axis = 1)

    xpid = tl.program_id(axis = 2)

    base = batchpid * batchstr + grouppid * groupstr

    row = xpid * MBLOCK + tl.arange(0 , MBLOCK)
    offsets = tl.arange(0 , KBLOCK)

    ptrx = base + xptr + row[: , None] * xstrm + offsets[None , :] * xstrn

    mean = 0.0
    count = 0

    for k in range(0 , K , KBLOCK):
        maskx = (row[: , None] < M) & ((k + offsets)[None , :] < K)

        x = tl.load(ptrx , mask = maskx , other = 0.0)

        mean += tl.sum(x)
        count += tl.sum(maskx.to(tl.int32))

        ptrx += KBLOCK * xstrn
    
    mean = mean / count
    ptrx = base + xptr + row[: , None] * xstrm + offsets[None , :] * xstrn

    var = 0.0
    for k in range(0 , K , KBLOCK):
        maskx = (row[: , None] < M) & ((k + offsets)[None , :] < K)

        x = tl.load(ptrx , mask = maskx , other = 0.0)

        diff  = tl.where(maskx, x - mean, 0.0)         
        var  += tl.sum(diff * diff)

        ptrx += KBLOCK * xstrn
    
    var = var / count
    inv_std = 1.0 / tl.sqrt(var + eps)
    ptrx = base + xptr + row[: , None] * xstrm + offsets[None , :] * xstrn
    ptr_out = base + outptr + row[:, None]*outstrm + offsets[None, :]*outstrn

    for k in range(0, K, KBLOCK):
        maskx = (row[:, None] < M) & ((k + offsets)[None, :] < K)
        x = tl.load(ptrx, mask=maskx, other=0.0)

        x = (x - mean) * inv_std
        

        tl.store(ptr_out, x, mask=maskx)
        ptrx += KBLOCK * xstrn
        ptr_out += KBLOCK * outstrn


    
def group_norm(x: torch.Tensor, num_groups: int, eps: float = 1e-5) -> torch.Tensor:

    N, C = x.shape[:2]
    spatial = x.numel() // (N * C)

    assert C % num_groups == 0, "C must be divisible by num_groups"
    channels_per_group = C // num_groups

    x_flat = x.contiguous().view(N, num_groups, channels_per_group * spatial)

    M = 1          
    K = channels_per_group * spatial

    out = torch.empty_like(x_flat)

    MBLOCK = 1
    KBLOCK = triton.next_power_of_2(min(K, 1024))

    grid = (N, num_groups, triton.cdiv(M, MBLOCK))

    groupnormkernel[grid](
        x_flat, out, eps,
        M, K,
        x_flat.stride(1), x_flat.stride(2),
        out.stride(1),    out.stride(2),
        x_flat.stride(0),                    # batchstr
        x_flat.stride(1),                    # groupstr
        MBLOCK=MBLOCK,
        KBLOCK=KBLOCK,
    )

    return out.view_as(x)
