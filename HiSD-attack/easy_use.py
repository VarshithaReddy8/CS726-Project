# import packages
from core.utils import get_config
from core.trainer import HiSD_Trainer
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from data_loader import get_loader
import copy
import torch.nn as nn

device = 'cpu'

# load checkpoint
config = get_config('configs/celeba-hq_256.yaml')
noise_dim = config['noise_dim']
image_size = config['new_size']
checkpoint = 'checkpoint_256_celeba-hq.pt'
trainer = HiSD_Trainer(config)
state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.to(device)

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


"""
DIY your translation steps.
e.g. change both 'Bangs' (latent-guided) and 'Eyeglasses' (reference-guided) to 'with'. 
"""

# Attribute in latent-guided:[0(bangs), 1(glasses)]
steps = [
    {'type': 'latent-guided', 'tag': 0, 'attribute': 1, 'seed': None},
    {'type': 'reference-guided', 'tag': 1, 'reference': 'examples/reference_glasses_0.jpg'}
]

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=5, a=0.1, feat = None):
        #k: iterations, a: step size
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y):
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()

        X.requires_grad = True
        X.retain_grad = True

        for i in range(self.k):
            # c_trg = translate(X)
            print(i)
            X.requires_grad = True
            X.retain_grad = True
            output = self.model(translate(X))

            loss = self.loss_fn(output, y)
            loss.backward()
            
            gradient = X.grad

            X_adv = X + self.a * gradient.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, X - X_nat

def translate(x):
    # with torch.no_grad():
    c_trg = E(x)
    for j in range(len(steps)):
        step = steps[j]
        if step['type']=='latent-guided':
            if step['seed'] is not None:
                torch.manual_seed(step['seed'])
                torch.cuda.manual_seed(step['seed'])
            z = torch.randn(1, noise_dim).to(device)
            s_trg = M(z, step['tag'], step['attribute'])

        elif step['type']=='reference-guided':
            reference = transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).to(device)
            s_trg = F(reference, step['tag'])

        c_trg = T(c_trg, s_trg, step['tag'])
    return c_trg



dataloader = get_loader(image_size=image_size)

def test():
    for i, (x_real, c_org) in enumerate(dataloader):
        c_trg = translate(x_real)
        with torch.no_grad():
            x_trg = G(c_trg)
        vutils.save_image(((x_trg + 1)/ 2).data, 'examples/output'+str(i)+'.jpg', padding=0)

def test_attack():
    for i, (x_real, c_org) in enumerate(dataloader):
        x_real = x_real.to(device)
        pgd_attack = LinfPGDAttack(model=G, device=device)

        x_real.requires_grad = True
        x_real.retain_grad = True
        # x_real_mod = x_real
        c_trg = translate(x_real)
        with torch.no_grad():
            y = G(c_trg)
        x_adv, perturb = pgd_attack.perturb(x_real, y)
        x_adv = x_real+perturb
        # print(perturb)
        c_adv = translate(x_adv)
        x_adv_trg = G(c_adv)
        vutils.save_image(((x_adv_trg + 1)/ 2).data, 'examples/output'+str(i)+'_attack.jpg', padding=0)
        vutils.save_image(((x_adv + 1)/ 2).data, 'examples/perturbed'+str(i)+'.jpg', padding=0)


def main(config):
    if config.mode=='test':
        test()
    elif config.mode=='test_attack':
        test_attack()
    else:
        print("Select mode")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test_attack', choices=['test', 'test_attack'])
    config = parser.parse_args()
    main(config)
