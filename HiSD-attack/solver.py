# import packages
from core.utils import get_config
from core.trainer import HiSD_Trainer
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.functional as F


class Solver(object):
    def __init__(self, dataloader, config):
        self.dataloader = dataloader
        self.reference_dir = config.reference_images
        self.results = config.results_dir
        self.device = 'cpu'
        self.config = get_config('configs/celeba-hq_256.yaml')
        self.noise_dim = self.config['noise_dim']
        self.image_size = self.config['new_size']
        self.checkpoint = config.checkpoint
        self.trainer = HiSD_Trainer(self.config)
        self.state_dict = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        self.trainer.models.gen.load_state_dict(self.state_dict['gen_test'])
        self.trainer.models.gen.to(self.device)

        self.c_dim = config.c_dim
        self.selected_attrs = config.selected_attrs

        self.E = self.trainer.models.gen.encode
        self.T = self.trainer.models.gen.translate
        self.G = self.trainer.models.gen.decode
        self.M = self.trainer.models.gen.map
        self.F = self.trainer.models.gen.extract

        self.seed = None

        self.transform = transforms.Compose([transforms.Resize(self.image_size), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #Attacks
        self.epsilon = 0.005
        self.k = 10
        self.a = 0.01
        self.loss_fn = nn.MSELoss().to(self.device)
        self.rand = True

    # PGD Attack
    def PGD(self, X_nat, y, c_org):
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()

        X.requires_grad = True
        X.retain_grad = True
        # epsilon = self.epsilon
        epsilon = 0.05

        for i in range(self.k):
            X.requires_grad = True
            X.retain_grad = True
            output = self.G(self.translate(X, c_org))

            loss = self.loss_fn(output, y)
            loss.backward()
            
            gradient = X.grad

            X_adv = X + self.a * gradient.sign()

            eta = torch.clamp(X_adv - X_nat, min=-epsilon, max=epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, X - X_nat

    # FGSM Attack
    def FGSM(self, X_nat, y, c_org):
        X = X_nat.clone().detach_()
        X.requires_grad = True
        X.retain_grad = True

        # epsilon = self.epsilon
        epsilon = 0.1
        output = self.G(self.translate(X, c_org))

        loss = self.loss_fn(output, y)
        loss.backward()
        
        gradient = X.grad

        X_adv = X + epsilon * gradient.sign()

        X = torch.clamp(X_adv, min=-1, max=1).detach()

        return X, X - X_nat

    # iFGSM Attack
    def iFGSM(self, X_nat, y, c_org):
        X = X_nat.clone().detach_()
        epsilon = self.epsilon
        for i in range(self.k):
            X.requires_grad = True
            X.retain_grad = True

            output = self.G(self.translate(X, c_org))

            loss = self.loss_fn(output, y)
            loss.backward()
            
            gradient = X.grad

            X_adv = X + epsilon * gradient.sign()

            X = torch.clamp(X_adv, min=-1, max=1).detach()

        return X, X - X_nat

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self):
        labels = []
        for i in range(self.c_dim):
            steps = []
            step = {'type': 'latent-guided', 'tag': 0, 'seed': self.seed}
            if(self.c_dim==1):
                if 'bangs' in self.selected_attrs:
                    step['attribute'] = 0
                if 'glasses' in self.selected_attrs:
                    step['attribute'] = 1
            else:
                step['attribute'] = i
            steps.append(step)
            if step['attribute'] == 0:
                reference = os.path.join(self.reference_dir, 'reference_bangs.jpg')
            else:
                reference = os.path.join(self.reference_dir, 'reference_glasses.jpg')
            step = {'type': 'reference-guided', 'tag': 1, 'reference': reference}
            steps.append(step)
            labels.append(steps)
        return labels

    def translate(self, x, steps):
        c_trg = self.E(x)
        for j in range(len(steps)):
            step = steps[j]
            if step['type']=='latent-guided':
                if step['seed'] is not None:
                    torch.manual_seed(step['seed'])
                    torch.cuda.manual_seed(step['seed'])
                z = torch.randn(1, self.noise_dim).to(self.device)
                s_trg = self.M(z, step['tag'], step['attribute'])

            elif step['type']=='reference-guided':
                reference = self.transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).to(self.device)
                s_trg = self.F(reference, step['tag'])

            c_trg = self.T(c_trg, s_trg, step['tag'])
        return c_trg

    def test(self):
        for i, (x_real, c_org) in enumerate(self.dataloader):
            x_list = [x_real]
            labels = self.create_labels()
            for j in range(len(labels)):
                c_trg = self.translate(x_real, labels[j])
                with torch.no_grad():
                    x_trg = self.G(c_trg)
                x_list.append(x_trg)
            x_concat = torch.cat(x_list, dim=3)
            result_path = os.path.join(self.results, 'image{}-trans.jpg'.format(i+1))
            vutils.save_image(self.denorm(x_concat.data.cpu()), result_path, padding=0)

    def test_attack(self, attack_type):
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(self.dataloader):
            labels = self.create_labels()
            x_fake_list = [x_real]
            for j in range(len(labels)):
                c_org = labels[j]
                x_real = x_real.to(self.device)

                x_real.requires_grad = True
                x_real.retain_grad = True
                c_trg = self.translate(x_real, c_org)
                with torch.no_grad():
                    y = self.G(c_trg)
                if attack_type=="PGD":
                    x_adv, perturb = self.PGD(x_real, y, c_org)
                elif attack_type=="FGSM":
                    x_adv, perturb = self.FGSM(x_real, y, c_org)
                elif attack_type=='iFGSM':
                    x_adv, perturb = self.iFGSM(x_real, y, c_org)
                x_adv = x_real+perturb
                c_adv = self.translate(x_adv, labels[j])
                if j==0:
                    x_fake_list.append(x_adv)
                with torch.no_grad():
                    x_adv_trg = self.G(c_adv)
                    x_fake_list.append(x_adv_trg)
                    l1_error += F.l1_loss(x_adv_trg, y)
                    l2_error += F.mse_loss(x_adv_trg, y)
                    l0_error += (x_adv_trg - y).norm(0)
                    min_dist += (x_adv_trg - y).norm(float('-inf'))
                    if F.mse_loss(x_adv_trg, y) > 0.05:
                        n_dist += 1
                    n_samples += 1
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.results, 'image{}-attacked.jpg'.format(i+1))
            vutils.save_image(self.denorm(x_concat.data.cpu()), result_path, padding=0)
            print('image', i, '- done')

        print('{} images. L1 error: {}. L2 error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples))
