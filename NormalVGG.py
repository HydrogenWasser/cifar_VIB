import math
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

class NormalVGG(nn.Module):

    """
    VGG16 VIB version
    """
    def __init__(self, device):
        super(NormalVGG, self).__init__()
        self.num_class = 10
        self.loss_func = nn.CrossEntropyLoss()
        self.device = device
        self.batch_size = 100
        #self.prior = Normal(torch.zeros(1, self.z_dim).to(self.device), torch.ones(1, self.z_dim).to(self.device))
        # conv1
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net

        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(in_features=512 * 7 * 7, out_features=1024),  # 自定义网络输入后的大小。
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=self.num_class)
        )

    def forward(self, x):   # output: 32 * 32 * 3
        final_pred, z_scores, features, outputs = self.causal_forward(x)
        return final_pred, z_scores, features, outputs
        # x (batchsize, 512)

    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    def batch_loss(self, logits, features, z_scores, y_batch):
        all_ces = []
        all_regs = []

        for i, logit in enumerate(logits):
            ce_loss = self.loss_func(logit, y_batch)
            # iter_info_print['ce_loss_{}'.format(i)] = ce_loss.sum().item()
            all_ces.append(ce_loss)

        for i in range(len(features)):
            reg_loss = self.smooth_l1_loss(features[i] * self.get_mean_wo_i(z_scores, i), self.get_mean_wo_i(features, i) * z_scores[i])
            # iter_info_print['ciiv_l1loss_{}'.format(i)] = reg_loss.sum().item()
            all_regs.append(reg_loss)

        loss = self.w_ce * sum(all_ces) / len(all_ces) + self.w_reg * sum(all_regs) / len(all_regs)

        return loss

    def create_mask(self, w, h, center_x, center_y, alpha=10.0):
        widths = torch.arange(w).view(1, -1).repeat(h,1)
        heights = torch.arange(h).view(-1, 1).repeat(1,w)
        mask = ((widths - center_x)**2 + (heights - center_y)**2).float().sqrt()
        # non-linear
        mask = (mask.max() - mask + alpha) ** 0.3
        mask = mask / mask.max()
        # sampling
        mask = (mask + mask.clone().uniform_(0, 1)) > 0.9
        mask.float()
        return mask.unsqueeze(0)

    def causal_forward(self, x):
        b, n, w, h = x.shape
        samples = []
        masks = []
        NUM_LOOP = 9
        NUM_INNER_SAMPLE = 3
        NUM_TOTAL_SAMPLE = NUM_LOOP * NUM_INNER_SAMPLE

        for i in range(NUM_TOTAL_SAMPLE):
            sample = self.relu(x + x.detach().clone().uniform_(-1, 1) * self.aug_weight)
            sample = sample / (sample + 1e-5)

            if i % NUM_INNER_SAMPLE == 0:
                idx = int(i // NUM_INNER_SAMPLE)
                x_idx = int(idx // 3)
                y_idx = int(idx % 3)
                center_x = self.mask_center[x_idx]
                center_y = self.mask_center[y_idx]

            mask = self.create_mask(w, h, center_x, center_y, alpha=10.0).to(x.device)
            sample = sample * mask.float()
            samples.append(sample)
            masks.append(mask)

        outputs = []
        features = []
        z_scores = []

        for i in range(NUM_LOOP):
            # Normalized input
            inputs = sum(samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
            z_score = (sum(masks[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]).float() / NUM_INNER_SAMPLE).mean()
            # forward modules

            feats = self.features(inputs)
            feats = feats.view(feats.size(0), -1)
            preds = self.classifier(feats)

            z_scores.append(z_score.view(1,1).repeat(b, 1))
            features.append(feats)
            outputs.append(preds)

        final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP

        return final_pred, z_scores, features, outputs


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
