import torch
import numpy as np
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets
import os

from maml_vae import MAML_VAE

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
batch_size = 128
epochs = 500
seed = 0
test_batch_size = 256
data_names = ["MNIST", "FashionMNIST"]
glength = 28
gwidth = 28

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def load_data(*datas):
    for data in datas:
        x = data.data.type(dtype=torch.float32) / 255.
        total_num, length, width = x.size()
        x = x.view(-1, length*width)
        # x = x[torch.randperm(total_num)]
        train_num = int(total_num * 0.8)
        x_train = x[:train_num]
        x_test = x[train_num:]
        yield x_train, x_test


if __name__ == "__main__":

    m_data = datasets.MNIST(root="./data/MNIST", train=True, download=False)
    f_data = datasets.FashionMNIST(root="./data/FashionMNIST", train=True, download=False)

    # data_train and data_test is tuple
    datas_train, datas_test = tuple(zip(*load_data(m_data, f_data)))
    datas_train = list(datas_train)
    datas_test = list(datas_test)
    
    # construct vae
    input_dim = datas_train[0].size(-1)

    Vae = MAML_VAE(input_dim, hidden_size=64, output_dim=4, data_names=data_names, device=device)

    # train
    Vae.train()
    train_nums = [data_train.size(0) for data_train in datas_train]
    for ep in range(epochs):
        # shuffle
        for idx, (data_train, train_num) in enumerate(zip(datas_train, train_nums)):
            datas_train[idx] = data_train[torch.randperm(train_num)]
        # record
        inner_losses_re = [0 for _ in train_nums]
        inner_losses_kl = [0 for _ in train_nums]
        outer_losses_re = [0 for _ in train_nums]
        outer_losses_kl = [0 for _ in train_nums]
        loss_num = 0
        for _ in range(max([train_num // batch_size for train_num in train_nums])):
            inner_indexs = [np.random.choice(train_num, batch_size, replace=False) for train_num in train_nums]
            outer_indexs = [np.random.choice(train_num, batch_size, replace=False) for train_num in train_nums]
            inner_loss_re, inner_loss_kl, outer_loss_re, outer_loss_kl = Vae.update(
                inner_x=[data_train[index] for data_train, index in zip(datas_train, inner_indexs)],
                outer_x=[data_train[index] for data_train, index in zip(datas_train, outer_indexs)]
            )
            loss_num += 1
            for idx, (inner_loss_re_i, inner_loss_kl_i, outer_loss_re_i, outer_loss_kl_i) in enumerate(zip(
                inner_loss_re, inner_loss_kl, outer_loss_re, outer_loss_kl
            )):
                inner_losses_re[idx] += inner_loss_re_i
                inner_losses_kl[idx] += inner_loss_kl_i
                outer_losses_re[idx] += outer_loss_re_i
                outer_losses_kl[idx] += outer_loss_kl_i

        inner_losses_re = [item / loss_num for item in inner_losses_re]
        inner_losses_kl = [item / loss_num for item in inner_losses_kl]
        outer_losses_re = [item / loss_num for item in outer_losses_re]
        outer_losses_kl = [item / loss_num for item in outer_losses_kl]
        print(f"Epoch: {ep}:")
        for data_name, ilr_i, ilk_i, olr_i, olk_i in zip(
            data_names, inner_losses_re, inner_losses_kl, outer_losses_re, outer_losses_kl
        ):
            print(f"\tData:{data_name}:")
            print(f"\t\tInnerReLoss: {ilr_i}")
            print(f"\t\tInnerKlLoss: {ilk_i}")
            print(f"\t\tOuterReLoss: {olr_i}")
            print(f"\t\tOuterKlLoss: {olk_i}")

    
    # test
    Vae.eval()
    z, _, _ = tuple(zip(*[Vae.encode(x_i) for x_i in datas_test]))
    x_re = [Vae.decode(z_i).clamp(0., 1.) for z_i in z]
    indexs = [np.random.choice(x_i.size(0), test_batch_size, replace=False) for x_i in datas_test]
    store_x_re = [x_re_i[index].view(-1, 1, glength, gwidth) for x_re_i, index in zip(x_re, indexs)]

    cat_data_name = ''
    for data_name in data_names:
        if cat_data_name == '':
            cat_data_name += data_name
        else:
            cat_data_name += ('-' + data_name)

    root_path = f'./image/maml/{cat_data_name}/vae/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for data_name, store_x_re_i in zip(data_names, store_x_re):
        torchvision.utils.save_image(store_x_re_i, os.path.join(root_path, f"{data_name}_test.png"), nrow=16)

    # generate
    x_gen = Vae.generate(num_sample=test_batch_size).clamp(0., 1.)
    x_gen = x_gen.view(-1, 1, glength, gwidth)
    torchvision.utils.save_image(x_gen, os.path.join(root_path, "generate.png"), nrow=16)