# python imports
import os
#import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    #img.SetOrigin(ref_img.GetOrigin())
    #img.SetDirection(ref_img.GetDirection())
    #img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin((0.0,0.0,0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, os.path.join(args.result_dir, name))




def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")


    # 读入图像
    # Get all the names of the training data
    vol_size = [128,256,256]#tensor shape = [N, C, D, W, H], vol_size指定D,W,H
    DS = Dataset(train_dir = args.train_dir,vol_size = vol_size)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Create DIR-Lung and STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()
    # number of the model parameters
    print("DIR-Lung: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Training loop.
    for i in range(1, args.n_iter + 1):
        # Generate the moving images and convert them to tensors.
        
        input_fixed,input_moving = iter(DL).next()     
        
        # [B, C, D, W, H]
        input_fixed = input_fixed.to(device).float()
        input_moving = input_moving.to(device).float()

        # Run the data through the model to produce warp and flow field
        flow_f2m = UNet(input_fixed,input_moving)
        f2m = STN(input_fixed,flow_f2m)

        # Calculate loss
        sim_loss = sim_loss_fn(f2m, input_moving)
        grad_loss = grad_loss_fn(flow_f2m)
        loss = sim_loss + args.alpha * grad_loss
        print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save images
            f_name = str(i) + "_f.nii.gz"
            m_name = str(i) + "_m.nii.gz"
            f2m_name = str(i) + "_f2m.nii.gz"
            save_image(input_fixed, f_name)
            save_image(input_moving, m_name)
            save_image(f2m, f2m_name)
            print("warped images have saved.")
            
        if i == args.n_iter:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
