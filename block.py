import torch
from attention import SelfAttention

class Block(torch.nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        layers = torch.nn.ModuleList([])
        for i in range(1):
            layers.extend([
                torch.nn.LayerNorm(dim),
                SelfAttention(dim, heads=heads, dim_head=dim_head),
                torch.nn.Linear(dim, dim),
                torch.nn.ELU(),
                torch.nn.Linear(dim, dim)
            ])
        self.layers = layers
    def forward(self, x):
        out = x
        for i in self.layers:
            out = i(out)

        return out

def test():
    block = Block(255, 8, 3, 10).cuda()
    print(sum(p.numel() for p in block.parameters() if p.requires_grad))
    print(block(torch.randn(1,30000,255).cuda()).shape)

if __name__ == '__main__':
    test()