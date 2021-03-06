import copy
import numpy as np

import torch
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, feat = None):
        #k: iterations, a: step size
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y, c_trg):
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()


        for i in range(self.k):
            X.requires_grad = True
            X.retain_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        return X, X - X_nat

class FGSM(object):
    def __init__(self, model=None, device=None, feat=None, epsilon=0.05):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.feat = feat
        self.loss_fn = nn.MSELoss().to(device)

    def perturb(self, X_nat, y, c_trg):
        X = X_nat.clone().detach()
        X.requires_grad = True
        X.retain_grad = True
        output, feats = self.model(X, c_trg)
        if self.feat:
            output = feats[self.feat]

        self.model.zero_grad()
        loss = self.loss_fn(output, y)
        loss.backward()
        grad = X.grad

        X_adv = X + self.epsilon*grad.sign()

        X = torch.clamp(X_adv, min=-1, max=1).detach()
        self.model.zero_grad()
        return X, X-X_nat

class iFGSM(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, feat = None):
        #k: iterations, a: step size
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

    def perturb(self, X_nat, y, c_trg):
        X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            X.retain_grad = True
            output, feats = self.model(X, c_trg)
            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.epsilon*grad.sign()

            X = torch.clamp(X_adv, min=-1, max=1).detach()

        return X, X - X_nat

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    adversary.model = model_cp

    X_adv, _ = adversary.perturb(X, y, c_trg)

    return X_adv