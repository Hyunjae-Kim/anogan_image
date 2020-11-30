import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torchvision import transforms, datasets
import torch.nn.init as init
from torchsummary import summary
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description = 'parameters')
parser.add_argument('--epoch', type=int, default=5001, help='# of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='# of mini-batch data')
parser.add_argument('--save_epoch_step', type=int, default=100, help='save model epoch')
parser.add_argument('--lat_num', type=int, default=100, help='# of latent variables z')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of GAN model')
parser.add_argument('--data_path', type=str, help='directory path of training dataset')
parser.add_argument('--ckpt_save_path', type=int, help='directory path of saving model parameters')
parser.add_argument('--img_save_path', type=int, help='directory path of saving generated image sample')

args = parser.parse_args()

def z_noise(lat_num, device):
    z_code = init.normal_(torch.Tensor(1,lat_num),mean=0,std=0.1)
    z = z_code.to(device)
    return z

epoch = args.epoch                  
batch_size = args.batch_size                
save_epoch_step = args.save_epoch_step        
lat_var_num = args.lat_num
learning_rate = args.lr         
data_dir_name = args.data_path      
ckpt_dir_name = args.ckpt_save_path         
img_dir_name = args.img_save_path           

print('\n\n training start !')

np.random.seed(37)
torch.manual_seed(37)
torch.cuda.manual_seed_all(37)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = nn.DataParallel(Generator(lat_num=lat_var_num)).to(device)
discriminator = nn.DataParallel(Discriminator()).to(device)

generator.train()
discriminator.train()

try:
    if not os.path.exists(ckpt_dir_name): 
        os.mkdir(ckpt_dir_name)
except FileExistsError:
    print('Already exist folder')

try:
    if not os.path.exists(img_dir_name):
        os.mkdir(img_dir_name)
except FileExistsError:
    print('Already exist folder')

train_set = datasets.ImageFolder(root=data_dir_name,
                       transform=transforms.Compose([
                         transforms.Grayscale(),
                         transforms.Resize((64,64)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    
train_loader = utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)

loss_func = nn.MSELoss()

G_optim = torch.optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5,0.999))
D_optim = torch.optim.Adam(discriminator.parameters(), lr = 4*learning_rate, betas=(0.5,0.999))

z_sample = z_noise(lat_var_num, device)

start_time = time.time()
gen_loss_list = np.array([])
dis_loss_list = np.array([])
for i in range(1,epoch):
    ones_label = torch.ones(batch_size,1).to(device)
    zeros_label = torch.zeros(batch_size,1).to(device)
    
    for j, x in enumerate(train_loader):
        x = torch.Tensor(x[0]).to(device)
        z_code = init.normal_(torch.Tensor(len(x),lat_var_num),mean=0,std=0.1).to(device)
        
        G_optim.zero_grad()
        G_fake = generator.forward(z_code)
        D_fake, _ = discriminator.forward(G_fake)
        G_loss = loss_func(D_fake, ones_label[:len(x)])
        G_loss.backward(retain_graph=True)
        G_optim.step()
        
        D_optim.zero_grad()
        z_code = init.normal_(torch.Tensor(len(x),lat_var_num),mean=0,std=0.1).to(device)
        G_fake = generator.forward(z_code)
        D_fake, _ = discriminator.forward(G_fake)
        D_real, _ = discriminator.forward(x)
        D_loss = loss_func(D_fake, zeros_label[:len(x)]) + loss_func(D_real, ones_label[:len(x)])
        D_loss.backward()
        D_optim.step()
    
    gen_loss_list = np.append(gen_loss_list, G_loss.cpu().detach().numpy())
    dis_loss_list = np.append(dis_loss_list, D_loss.cpu().detach().numpy())    
    
    if i%10 == 0:
        print("{}th epoch - gen_loss: {} dis_loss: {}".format(i, G_loss.data, D_loss.data))
        
    if i%save_epoch_step == 0:
        print('save model : %d th epoch'%i)
        torch.save(generator.state_dict(),'%s/generator_%d.pkl'%(ckpt_dir_name, i))
        torch.save(discriminator.state_dict(),'%s/discriminator_%d.pkl'%(ckpt_dir_name, i))
        
        gen_sample = generator.forward(z_sample).cpu()
        plt.imshow(gen_sample.data.numpy()[0][0], cmap='gray')
        plt.savefig('%s/sample_%d.png'%(img_dir_name, i))
        plt.clf()
        
        np.save('%s/gen_loss.npy'%ckpt_dir_name, gen_loss_list)
        np.save('%s/dis_loss.npy'%ckpt_dir_name, dis_loss_list)

running_time = round(time.time() - start_time, 4)
print('\n\n training finish! : time = {}'.format(running_time))