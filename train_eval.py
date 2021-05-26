import torch
import glob
import os
from eye_noise import filter_eye_noise
from vision_transformer import ImageTransformer
import random
import time
import torch.nn.functional as F

random.seed(1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_path = '/Users/a18651298/PycharmProjects/hackathon/'
        self.data = []
        file_list = glob.glob(self.data_path + "word_*")
        for class_path in file_list:
            label = int(class_path.split("_")[-1]) - 1
            subfile_list = os.listdir(class_path)
            subfile_list = [os.path.join(class_path, word_num) for word_num in subfile_list]
            for tensor_path in subfile_list:
                self.data.append([tensor_path, label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word_path, label = self.data[index]
        tensor = torch.load(word_path)
        tensor = filter_eye_noise(tensor)
        return tensor.float(), label


def evaluate(model, data_loader, loss_history, test_name):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage ' + test_name + ' loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        print('Processing batch ' + str(i+1))
        optimizer.zero_grad()
        output = F.log_softmax(model(data.double()), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


if __name__ == '__main__':
    N_EPOCHS = 1000
    params = {'batch_size': 100,
              'shuffle': True,
              'num_workers': 2}

    full_dataset = Dataset()

    training_set, test_set = torch.utils.data.random_split(full_dataset,
                                                           [int(0.9 * len(full_dataset)),
                                                            int(0.1 * len(full_dataset)) + 1])
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    test_loader = torch.utils.data.DataLoader(test_set, **params)

    nn_model = ImageTransformer(num_classes=9, dim=1024, depth=6, heads=1, mlp_dim=128)
    nn_model = nn_model.double()
    nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train(nn_model, nn_optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        evaluate(nn_model, train_loader, test_loss_history, 'train')
        evaluate(nn_model, test_loader, test_loss_history, 'validation')

    print('Execution time')

    PATH = "trained_net.pt"
    torch.save(nn_model.state_dict(), PATH)
