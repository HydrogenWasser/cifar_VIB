from utils import *


"                                                     "
"               Los Geht's!                           "
"                                                     "


num_epoch = 50
HydrogenIB_train_tag = False
cnn_trian_tag = False
HydrogenIB_attac_tag = True
cnn_attack_tag = False


model = HydrogenIB(device=device).to(device)
cnn_model = CleanCNN(device=device).to(device)
ema = EMA(0.999)
if HydrogenIB_train_tag:
    HydrogenIB_Train(model=model, ema=ema, num_epoch=num_epoch)
    HydrogenIB_eval(model=model)
if cnn_trian_tag:
    Clean_CNN_Train(model=cnn_model, num_epoch=num_epoch)
if HydrogenIB_attac_tag:
    HydrogenIB_fgsm(model, 0.031)
    HydrogenIB_fgsm(model, 0.1)
    HydrogenIB_fgsm(model, 0.2)
    HydrogenIB_fgsm(model, 0.3)
    HydrogenIB_fgsm(model, 0.4)
# if cnn_attack_tag:
#     clean_CNN_foolbox_attack(cnn_model, 0.045)
