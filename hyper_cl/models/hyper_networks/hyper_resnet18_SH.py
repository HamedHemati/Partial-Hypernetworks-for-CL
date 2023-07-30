import torch
import torch.nn as nn
import functorch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d

from avalanche.models.dynamic_modules import MultiTaskModule


#########################################################
#                      ResNet-18
#########################################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes,
                               track_running_stats=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2],
                 num_classes=10, nf=20) -> None:
        super(ResNet18, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.classifier = nn.Linear(160, num_classes)

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @property
    def n_params(self):
        return [sum([p.shape.numel() for p in self.parameters()])]

    @property
    def param_shapes(self):
        return [p.shape.numel() for p in self.parameters()]

    @property
    def param_shapes_origin(self):
        return [p.shape for p in self.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


#########################################################
#                      HyperNetwork
#########################################################

class GenHead(nn.Module):
    def __init__(self, origin_shape, inp_dim=20, emb_dim=32):
        super().__init__()
        self.origin_shape = origin_shape
        self.emb_dim = emb_dim

        # Embedding for the first channel
        self.n_embs1 = origin_shape[0]
        self.emb1 = nn.Embedding(
            num_embeddings=self.n_embs1,
            embedding_dim=emb_dim
        )
        self.register_buffer("emb1_inp", torch.arange(0, self.n_embs1))

        # Linear transformation
        self.linear = nn.Linear(inp_dim + emb_dim, origin_shape[1:].numel())

        self.tanh = nn.Tanh()

    @property
    def n_params(self):
        return [sum([p.shape.numel() for p in self.parameters()])]

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.n_embs1, 1)
        embs1 = self.emb1(self.emb1_inp).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x_cat = torch.cat([x, embs1], dim=2)
        out = self.linear(x_cat)
        out = out.view(x.shape[0], *self.origin_shape)

        return out


class LinHead(nn.Module):
    def __init__(self, n_tasks, dim, value=1.0):
        super().__init__()
        self.params = nn.Parameter(torch.empty(n_tasks, dim).fill_(value))

    @property
    def n_params(self):
        return [sum([p.shape.numel() for p in self.parameters()])]

    def forward(self, idx):
        return self.params[idx]


class LinearLayer(nn.Module):
    def __init__(self, inp_size, out_shape: torch.Size):
        super().__init__()
        self.out_shape = out_shape
        self.linear = nn.Linear(inp_size, out_shape.numel())

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.shape[0], *self.out_shape)
        return out


class WeightGenerator(nn.Module):
    def __init__(self,
                 main_model,
                 num_tasks=10,
                 embd_dim=8,
                 hidden_size_1=50,
                 hidden_size_2=50,
                 head_emb_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embd_dim)

        self.layers = nn.Sequential(
            nn.Linear(embd_dim, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList()
        for n, p in main_model.named_parameters():
            shape = p.shape

            if len(shape) == 4:
                self.heads.append(
                    GenHead(shape, inp_dim=hidden_size_2, emb_dim=head_emb_dim))
            else:
                if n.startswith(("linear", "classifier")):
                    if n.endswith(".weight"):
                        self.heads.append(
                            LinearLayer(inp_size=hidden_size_2,
                                        out_shape=p.shape)
                        )
                    elif n.endswith(".bias"):
                        self.heads.append(
                            LinHead(num_tasks, shape.numel(), value=0.0))
                else:
                    if n.endswith(".bias"):
                        self.heads.append(
                            LinHead(num_tasks, shape.numel(), value=0.0))
                    elif n.endswith(".weight"):
                        self.heads.append(
                            LinHead(num_tasks, shape.numel(), value=1.0))
                    else:
                        raise NotImplementedError()

    @property
    def n_params(self):
        total = 0
        for p in self.parameters():
            total += p.shape.numel()
        return total

    def head_forward(self, head, x, idx):
        if isinstance(head, GenHead) or isinstance(head, LinearLayer):
            return head(x)
        elif isinstance(head, LinHead):
            return head(idx)

    def forward(self, idx):
        emb = self.embedding(idx)
        x = self.layers(emb)
        outs = [self.head_forward(head, x, idx) for head in self.heads]

        return outs


#########################################################
#                     HyperResNet
#########################################################

class EmptyMapping(nn.Module):
    def __init__(self):
        super(EmptyMapping, self).__init__()

    def forward(self, x):
        return x


class HyperResNet18SH(MultiTaskModule, nn.Module):
    def __init__(self, num_tasks=10, num_classes=10, embd_dim=32,
                 hidden_size_1=100,
                 hidden_size_2=50, head_emb_dim=32):
        super().__init__()

        # Main network
        self.feat_extractor_sf = EmptyMapping()
        self.feat_extractor_sl = ResNet18(num_classes=num_classes)
        for p in self.feat_extractor_sl.parameters():
            p.requires_grad = False

        # Weight generator
        self.weight_generator = WeightGenerator(self.feat_extractor_sl,
                                                num_tasks=num_tasks,
                                                embd_dim=embd_dim,
                                                hidden_size_1=hidden_size_1,
                                                hidden_size_2=hidden_size_2,
                                                head_emb_dim=head_emb_dim)

        # Functionals
        mn_func, mn_params = functorch.make_functional(self.feat_extractor_sl)
        self.mn_func_vmap = functorch.vmap(mn_func)

    def forward(self, x, task_ids):
        x = self.feat_extractor_sf(x)

        x = x.unsqueeze(1)
        gen_params = self.weight_generator(task_ids)
        out = self.mn_func_vmap(gen_params, x)
        out = out.squeeze(1)

        return out


if __name__ == "__main__":
    model = HyperResNet18SH(num_tasks=10,
                            num_classes=10,
                            embd_dim=32,
                            hidden_size_1=50,
                            hidden_size_2=32,
                            head_emb_dim=32)
    t = torch.LongTensor([0, 0, 0, 0, 0])
    x = torch.randn(5, 3, 32, 32)
    model(x, t)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model(x.to(device), t.to(device))

    print()
