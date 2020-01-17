import gym
import torch
from torch import nn
from window import Window

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.linear = nn.Linear(1452, 1452)
        self.weight = nn.Parameter(torch.randn((3, 3, 45, 45)), requires_grad=True)

    def forward(self, x):
        x = self.linear(x.flatten())
        x = x.reshape((1, 3, 22, 22))
        x = torch.conv_transpose2d(x, self.weight, stride=1, padding=1)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=3, padding=1)

        self.keys = nn.ParameterList(
            [nn.Parameter(torch.randn((1, 3, 22, 22)), requires_grad=True) for x in range(10)])
        self.modulelist = nn.ModuleList(Module() for x in range(10))

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)

        # Find most similar feature map in keys, use it's index to get the module from modulelist
        args = [torch.cosine_similarity(x.flatten(), key.flatten(), dim=0) for key in self.keys]
        argmax = torch.argmax(torch.Tensor(args))
        x = self.modulelist[argmax](x)
        x = torch.sigmoid(x)
        return x


try:
    env = gym.make("procgen:procgen-coinrun-v0")
    net = Model()
    window = Window()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    for i_episode in range(20000):
        gym.make("procgen:procgen-coinrun-v0")
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            x_obs = torch.Tensor([observation]).permute(0, 3, 1, 2) / 255
            out = net(x_obs).permute(0, 2, 3, 1)

            x_obs = torch.Tensor([observation]) / 255
            loss = criterion(out, x_obs)
            # loss = 1/torch.cosine_similarity(out.flatten(), x_obs.flatten(), dim=0) - 1
            loss.backward()
            print(f'LOSS: {loss.detach()}')
            optimizer.step()
            optimizer.zero_grad()
            out = out.detach().squeeze().numpy()
            window.imshow(out)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
finally:
    env.close()
