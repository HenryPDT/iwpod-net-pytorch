import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from iwpod.model import IWPODNet
from iwpod.dataset import ALPRDataset
from iwpod.loss import iwpodnet_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--model-dir', type=str, default='weights', help='Directory containing models and weights')
    parser.add_argument('-cm', '--cur_model', type=str, default='fake_name', help='Pre-trained model')
    parser.add_argument('-n', '--name', type=str, default='iwpodnet_retrained', help='Output model name')
    parser.add_argument('-tr', '--train-dir', type=str, default='train_dir', help='Input data directory for training')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='Number of epochs (default = 1,500)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='Mini-batch size (default = 64)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate (default = 0.001)')
    parser.add_argument('-se', '--save-epochs', type=int, default=2000, help='Freqnecy for saving checkpoints (in epochs) ')
    args = parser.parse_args()

    MaxEpochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_epochs = args.save_epochs

    netname = args.name
    train_dir = args.train_dir
    modeldir = args.model_dir

    modelname = '%s/%s' % (modeldir, args.cur_model)

    dim = 208
    mymodel = IWPODNet()
    opt = optim.Adam(mymodel.parameters(), lr=learning_rate)

    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading training data...')

    train_dataset = ALPRDataset(train_dir, dim=dim)
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=use_cuda,
        persistent_workers=False,
    )

    mymodel.train()
    mymodel.to(device)

    epoch_last = 0
    # checkpoint = torch.load('./weights/iwpodnet_retrained_epoch12000.pth')
    # mymodel.load_state_dict(checkpoint['model_state_dict'])
    # opt.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_last = checkpoint['epoch'] + 1

    for epoch in range(epoch_last,MaxEpochs):
        cost = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = mymodel(inputs)
            loss = iwpodnet_loss(labels, outputs)
            loss.mean().backward()
            opt.step()

            cost += loss.mean().item()

        cost = cost / len(train_loader)
        print(f"Epoch {epoch + 1}/{MaxEpochs} Loss: {cost:.4f}")

        if (epoch + 1) % save_epochs == 0:
            model_path_ckpt = os.path.join(modeldir, netname + '_epoch%d' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': mymodel.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'cost': cost
            }, model_path_ckpt + '.pth')

    print('Finished training the model')

    # Save the trained model
    model_path_final = os.path.join(modeldir, netname)
    torch.save({'model_state_dict': mymodel.state_dict()}, model_path_final + '.pth')
    print('Saved model at:', model_path_final + '.pth')