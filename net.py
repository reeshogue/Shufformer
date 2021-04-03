import torch
import torchvision as Tv
from block import Block
import random

class ShuffleAttention2d(torch.nn.Module):
    def __init__(self, C=32*32):
        super().__init__()
        self.C = C
        self.block = Block(C, 3, 64)
    def forward(self, x):
        x_shape = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=-1).unsqueeze(0)
        x = torch.split(x, self.C, -1)
        attention = []
        for i in x:
            attended = self.block(i)
            attention.append(attended)
        attention = torch.stack(attention, 1)
        attention = torch.transpose(attention, 1, 2).contiguous()
        attention = attention.view(*x_shape)
        return attention
        


class TestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ShuffleAttention2d()
        self.block2 = ShuffleAttention2d()
        self.block3 = ShuffleAttention2d()
        self.block4 = ShuffleAttention2d()
        self.output_head = torch.nn.Linear(32*32*3, 10)
    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1).unsqueeze(1)
        y = self.block(x) + x
        y = self.block2(y) + y
        y = self.block3(y) + y
        y = self.block4(y) + y
        y = torch.flatten(y, 1, -1)
        y = self.output_head(y)
        y = torch.nn.functional.log_softmax(y, dim=-1)
        return y

def test_train():
    net = TestNet().cuda()
    optim = torch.optim.Adam(net.parameters(), lr=1e-5)
    transform = Tv.transforms.Compose(
        [
            Tv.transforms.ToTensor()
        ]
    )

    
    trainset = Tv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)

    testset = Tv.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)

    data = []
    for i in range(1000):
        data.append(trainset[i])

    loss = torch.nn.CrossEntropyLoss()

    print(len(trainloader))

    datarand = random.sample(data, len(data))

    for epoch in range(1):
        for i in range(1000):
            data = datarand[i]
            inputs, labels = data
            
            inputs = inputs.unsqueeze(0).cuda()
            labels = labels
            optim.zero_grad()

            outputs = net(inputs)
            label = torch.zeros_like(outputs)
            for j in range(len(label)):
                label[j][labels] = 1.0
            losses = -torch.sum(label * outputs)
            losses.backward()
            optim.step()
            print(i, losses)

    correct = 0
    total = 0

    data = []
    for i in range(1000):
        data.append(testset[i])

    datarand = random.sample(data, len(data))

    with torch.no_grad():
        for i in range(1000):

            data = datarand[i]
            images, labels = data
            images = images.unsqueeze(0)
            outputs = net(images.cuda()).cpu()
            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (predicted == labels).sum().item()
    print(100*correct/total)

if __name__ == '__main__':
    test_train()

