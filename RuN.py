from utils import *


"                                                     "
"               Los Geht's!                           "
"                                                     "


num_epoch = 50
HydrogenIB_train_tag = False
cnn_trian_tag = True
HydrogenIB_attac_tag = False
cnn_attack_tag = False


model = resnetIB(device=device).to(device)
normal_model = resnet(device=device).to(device)
ema = EMA(0.999)
normal_ema = EMA(0.999)
if HydrogenIB_train_tag:
    resnetIB_Train(model=model, ema=ema, num_epoch=num_epoch)
    resnetIB_eval(model=model)
if cnn_trian_tag:
    resnet_Train(model=normal_model,ema=normal_ema, num_epoch=num_epoch)
# if HydrogenIB_attac_tag:
#     HydrogenIB_fgsm(model, 0.031)
#     HydrogenIB_fgsm(model, 0.1)
#     HydrogenIB_fgsm(model, 0.2)
#     HydrogenIB_fgsm(model, 0.3)
#     HydrogenIB_fgsm(model, 0.4)
# if cnn_attack_tag:
#     clean_CNN_foolbox_attack(cnn_model, 0.045)
