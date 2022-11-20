import torch
from tqdm import *
import numpy as np
import torchvision
from torchvision import datasets

from vae import CVAE, CVAE2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
batch_size = 128
epochs = 100
seed = 0
test_batch_size = 256
is_cvae2 = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":

    # all_data = datasets.MNIST(root="./data/MNIST", train=True, download=False)
    all_data = datasets.FashionMNIST(root="./data/FashionMNIST", train=True, download=False)
    
    x = all_data.data.type(dtype=torch.float32) / 255.
    label_dim = len(all_data.classes)
    y = torch.eye(label_dim)[all_data.targets] #one hot label
    total_num, length, width = x.size()
    x = x.view(-1, length*width)
    _, input_dim = x.size()
    # shuffle x
    randidx = torch.randperm(total_num)
    x = x[randidx]
    y = y[randidx]
    # train and test set
    train_num = int(total_num * 0.8)
    x_train = x[:train_num]
    x_test = x[train_num:]
    y_train = y[:train_num]
    y_test = y[train_num:]

    test_num , _ = x_test.size()
    # construct vae
    if not is_cvae2:
        Vae = CVAE(input_dim, hidden_size=64, output_dim=4, label_dim=label_dim, device=device)
    else:
        Vae = CVAE2(input_dim, hidden_size=64, output_dim=4, label_dim=label_dim, device=device)

    # train
    Vae.train()
    for ep in range(epochs):
        losses = 0
        losses_kl = 0
        loss_num = 0
        print(f"===================================EPOCH: {ep}===================================")
        for i in tqdm(range(0, train_num, batch_size)):
            indexs = list(range(i , i + batch_size)) if i + batch_size <= train_num else list(range(i , train_num))
            loss, loss_kl = Vae.update(x_train[indexs], y_train[indexs])
            losses += loss
            losses_kl += loss_kl
            loss_num += 1
        
        print(f"Epoch: {ep} | Loss: {losses / loss_num} | Loss_Kl: {losses_kl / loss_num}")

    # test
    Vae.eval()
    if not is_cvae2:
        z, _, _, y_mus = Vae.encode(x_test, y_test)
        x_test_re = Vae.decode(z, y_mus).clamp(0., 1.)
    else:
        z, _, _ = Vae.encode(x_test, y_test)
        x_test_re = Vae.decode(z, y_test).clamp(0., 1.)
    indexs = np.random.choice(test_num, test_batch_size, replace=False)
    store_x_re = x_test_re[indexs].view(-1, 1, length, width)
    
    save_idx = 2 if is_cvae2 else 1
    torchvision.utils.save_image(store_x_re, f"./image/fashionmnist/cvae/cvae{save_idx}/test.png", nrow=16)

    # generate
    x_gen = Vae.generate(num_sample=test_batch_size).clamp(0., 1.)
    x_gen = x_gen.view(label_dim, -1, 1, length, width) #(label_dim, num_sample, input_dim)
    for idx, x_gen_label in enumerate(torch.unbind(x_gen)):
        torchvision.utils.save_image(x_gen_label, f"./image/fashionmnist/cvae/cvae{save_idx}/generate_{idx}.png", nrow=16)