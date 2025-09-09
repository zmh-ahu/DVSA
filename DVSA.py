import torch
import torch.nn.functional as F
from torch import softmax
def DVSA_selective_attention_with_full_attention(q, k, v, top_u, w):
    """
        DVSA*:select the largest column by computing the complete attention weight map, corresponding to DVSA*-T in the paper.

        Parameters:
        - top_u (int): Control the number of highest value columns.
        - w (int): Focused context width of the self-attention.

        Returns:
        - torch.Tensor: Returns an attention weight matrix containing only some columns and diagonal values(the rest are assigned to negative infinity).
    """
    w = w + 1
    attention_weights = torch.bmm(q, k.transpose(1, 2))

    attention_sums = torch.sum(attention_weights, dim=1)

    sum_topk = torch.topk(attention_sums, top_u, sorted=False)

    mask = torch.ones_like(attention_weights)
    if w > 0:
        upper_tri_mask = torch.triu(torch.ones_like(mask), diagonal=w)
        lower_tri_mask = 1 - torch.triu(mask, -w+1)
        mask = upper_tri_mask + lower_tri_mask

    for i in range(mask.size(0)):
        mask[i, :, sum_topk.indices[i]] = 0

    boolean_mask = mask == 1

    attention_weights_DVSA = attention_weights.masked_fill(boolean_mask, float('-inf'))
    attention_weights_DVSA = softmax(attention_weights_DVSA, dim=2)

    out = torch.bmm(attention_weights_DVSA, v)

    return out, attention_weights_DVSA


def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max
def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

def DVSA_selective_attention_with_local_attention(q, k, v, top_u, w):
    """
        DVSA:select the largest column by computing the local attention weight map approximately, corresponding to DVSA-T(see fig.4) in the paper.

        Parameters:
        - top_u (int): Control the number of highest value columns.
        - w (int): Focused context width of the self-attention.

        Returns:
        - torch.Tensor: Returns an attention weight matrix containing only some columns and diagonal values(the rest are assigned to negative infinity).
    """
    look_forward = look_backward = w
    window_size = 1

    merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
    q, k, v = map(merge_into_batch, (q, k, v))

    b, t, e, device, dtype = *q.shape, q.device, q.dtype
    assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

    #Computing local attention
    windows = t // window_size

    ticker = torch.arange(t, device=device, dtype=torch.long)[None, :]  # tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    b_t = ticker.reshape(1, windows, window_size)

    bucket_fn = lambda t: t.reshape(b, windows, window_size, -1)
    bq, bk, bv = map(bucket_fn, (q, k, v))

    look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
    bk = look_around(bk, **look_around_kwargs)
    bq_k = look_around(b_t, **look_around_kwargs)

    dots = torch.einsum('bhie,bhje->bhij', bq, bk)

    mask_value = max_neg_value(dots)

    mask = bq_k[:, :, None, :] == -1
    dots.masked_fill_(mask, 0)
    del mask

    dots = dots.squeeze(2)
    B, M, L = dots.size()
    dots = F.pad(dots, (0, M - 1), value=0)
    dots = dots.view(B, -1)
    dots = dots[:, :-M]  # B x ML+MM
    dots = dots.view(B, M, M + L - 1 - 1)
    dots = dots[:, :, look_backward:-look_forward + 1]#Get the local attention

    # Extract the maximum column based on local attention and calculate the vertical part
    if dots.size(1) >= 2*look_forward+1:
        mul = []
        n=1
        for i in range(dots.size(1)):
            if i < look_forward:
                mul.append((2*look_forward+1)/(look_forward+i+1))
            elif i >= dots.size(1)-look_forward:
                mul.append((2 * look_forward + 1) / ((2 * look_forward + 1)-n))
                n=n+1
            else:
                mul.append(1)
    mul = torch.tensor(mul)
    mul = mul.unsqueeze(0)
    dots_sum = torch.sum(dots*mul, dim=1)

    M_top = torch.topk(dots_sum, top_u, sorted=False)[1].squeeze(0)
    K_reduce = k[torch.arange(B)[:, None], M_top, :]

    Q_K = torch.matmul(q, K_reduce.transpose(-2, -1))
    dots[torch.arange(B)[:, None], :, M_top] = Q_K.transpose(1, 2)

    mask = dots[:, :, :] == 0
    dots.masked_fill_(mask, mask_value)
    attention_weights_DVSA = dots
    del mask

    attention_weights_DVSA = softmax(attention_weights_DVSA, dim=-1)

    out = torch.bmm(attention_weights_DVSA, v)

    return out, attention_weights_DVSA

q = k = v = torch.tensor([[[1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1., 1., 1., 1.]]])
top_u = 1
w = 2

out1, attention_weights_DVSA1 = DVSA_selective_attention_with_full_attention(q, k, v, top_u, w)# DVSA*
out2, attention_weights_DVSA2 = DVSA_selective_attention_with_local_attention(q, k, v, top_u, w)# DVSA

#Print attention weight map
print(attention_weights_DVSA1)
# tensor([[[0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
#          [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
#          [0.1667, 0.0000, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000],
#          [0.1667, 0.0000, 0.0000, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
#          [0.2000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000],
#          [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500]]])
print(attention_weights_DVSA2)
# tensor([[[0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
#          [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
#          [0.1667, 0.0000, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000],
#          [0.1667, 0.0000, 0.0000, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
#          [0.2000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000],
#          [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500]]])
