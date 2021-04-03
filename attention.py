import torch
import einops
#Modified code from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py.
#Original 'Performer' paper: https://arxiv.org/abs/2009.14794
#Added comments and stuff to make the code more readable and for better maintainability.


def orthogonal_gaussian_matrix_chunk(columns, device = None):
    #Takes a gaussian sample and returns the Q in the QR factorization of the sample.
    #QR factor]ization is defined as: input = inner(q, r), where Q is defined to be the orthogonal we want, and 
    #R is defined to be a lower triangular matrix which we don't need...
    #Now, what is an orthogonal matrix? An orthogonal matrix is defined as: matmul(Q, transpose(Q)) = the identity matrix...

    block = torch.randn((columns, columns), device=device)
    q, _ = torch.linalg.qr(block, mode='reduced')
    return q.t()

def orthogonal_gaussian_matrix(num_rows, num_cols, device = None):
    num_blocks = int(num_rows/num_cols)
    
    blocks = []
    for _ in range(num_blocks):
        chunk = orthogonal_gaussian_matrix_chunk(num_cols, device=device)
        blocks.append(chunk)
    
    remaining_rows = num_rows - num_blocks * num_cols
    
    if remaining_rows > 0:
        chunk = orthogonal_gaussian_matrix_chunk(num_cols, device=device)
        blocks.append(chunk[:remaining_rows])
    
    matrix = torch.cat(blocks)
    multiplier = torch.randn((num_rows,), device=device)
    
    return torch.diag(multiplier) @ matrix

def generalized_kernel(data, projection_matrix, kernel_fn, epsilon=1e-6):
    b, h, *_ = data.shape

    projection_matrix = einops.repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    data_dash = torch.einsum('...id,...jd->...ij', data, projection_matrix)

    data_prime = kernel_fn(data_dash) + epsilon

    return data_prime

def linear_attention(q, k, v):
    k_sum = k.sum(dim=-2)
    d_inv = 1. / torch.einsum('...nd,...d->...n', q, k_sum)
    contexts = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne',contexts, q, d_inv)
    return out

class GeneralizedAttention(torch.nn.Module):
    def __init__(self, head_dims, num_features):
        super().__init__()
        self.head_dims = head_dims
        self.num_features = num_features

        projection_matrix = orthogonal_gaussian_matrix(num_features, head_dims)
        self.register_buffer('proj_mat', projection_matrix)

        self.kernel_fn = torch.nn.GELU()

    def forward(self, q, k, v):
        kernel = lambda x: generalized_kernel(x, self.proj_mat, self.kernel_fn)
        q = kernel(q)
        k = kernel(k)

        out = linear_attention(q, k, v)
        return out

class SelfAttention(torch.nn.Module):
    def __init__(self, dim, heads=2, dim_head=64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.gen_attn = GeneralizedAttention(dim_head, dim_head)
        self.to_q = torch.nn.Linear(dim, inner_dim)
        self.to_k = torch.nn.Linear(dim, inner_dim)
        self.to_v = torch.nn.Linear(dim, inner_dim)
        self.to_out = torch.nn.Linear(inner_dim, dim)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda y: einops.rearrange(y, 'b n (h d) -> b h n d', h=h), (q, k, v))
        out = self.gen_attn(q, k, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out