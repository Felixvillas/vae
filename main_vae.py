import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets

from vae import VAE

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
batch_size = 128
epochs = 5000
seed = 0
test_batch_size = 256

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":

    all_data = datasets.MNIST(root="./data/MNIST", train=True, download=False)
    
    x = all_data.data.type(dtype=torch.float32) / 255.
    total_num, length, width = x.size()
    x = x.view(-1, length*width)
    _, input_dim = x.size()
    # shuffle x
    x = x[torch.randperm(total_num)]
    # train and test set
    train_num = int(total_num * 0.8)
    x_train = x[:train_num]
    x_test = x[train_num:]
    test_num , _ = x_test.size()
    # construct vae
    Vae = VAE(input_dim, hidden_size=64, output_dim=4, device=device)

    # train
    Vae.train()
    for ep in range(epochs):
        indexs = np.random.choice(train_num, batch_size, replace=False)
        loss, loss_kl = Vae.update(x_train[indexs])
        print(f"Epoch: {ep} | Loss: {loss} | Loss_Kl: {loss_kl}")

    # test
    Vae.eval()
    z, _, _ = Vae.encode(x_test)
    x_test_re = Vae.decode(z).clamp(0., 1.)
    indexs = np.random.choice(test_num, test_batch_size, replace=False)
    store_x_re = x_test_re[indexs].view(-1, 1, length, width)
    torchvision.utils.save_image(store_x_re, f"./image/mnist/vae/test.png", nrow=16)

    # generate
    x_gen = Vae.generate(num_sample=test_batch_size).clamp(0., 1.)
    x_gen = x_gen.view(-1, 1, length, width)
    torchvision.utils.save_image(x_gen, f"./image/mnist/vae/generate.png", nrow=16)