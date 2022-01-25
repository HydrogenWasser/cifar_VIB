from cifar_utils import *


"                                                     "
"               Los Geht's!                           "
"                                                     "


num_epoch = 200
causalVGG_train_tag = True
normalVGG_trian_tag = False
causalVGG_attac_tag = True
normalVGG_attack_tag = False

# betas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]
betas = [0.01]
epsilon = [0.031, 0.045, 0.083]
model = VGG_VIB(device=device).to(device)
normal_model = NormalVGG(device=device).to(device)
ema = EMA(0.999)
normalEma = EMA(0.999)

if causalVGG_train_tag:
    for beta in betas:
        print("---------", beta, "------------")
        causalVGG_Train(beta, model, ema, num_epoch)


if normalVGG_trian_tag:
    NormalVGG_Train(normal_model,normalEma, num_epoch)


if causalVGG_attac_tag:
    for beta in betas:
        print("---------", beta, "------------")
        for eps in epsilon:
            causalVGG_eval(beta, model)
            causalVGG_adver(beta, model, eps)
if normalVGG_attack_tag:
    NormalVGG_eval(normal_model)
    for eps in epsilon:
        normalVGG_adver(normal_model, eps)


