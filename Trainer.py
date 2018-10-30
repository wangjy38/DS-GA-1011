import pickle as pkl
import torch
from SNLI_dataset_loader import SNLI_Dataset, SNLI_collate_func
from models.CNN import CNN
from models.RNN import RNN
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np


def read_data(filepath):

    processed_snli_data_train = pkl.load(
        open(filepath + "processed_snli_data_train.p", "rb"))
    processed_snli_data_val = pkl.load(
        open(filepath + "processed_snli_data_val.p", "rb"))
    loaded_embeddings_ft = pkl.load(
        open(filepath + "loaded_embeddings_ft.p", "rb"))

    return processed_snli_data_train, processed_snli_data_val, loaded_embeddings_ft


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for sentence_1, length_1, sentence_2, length_2, labels in loader:
        sentence_1_batch, length_1_batch, sentence_2_batch, length_2_batch, labels =\
            sentence_1.to(device), length_1.to(device), sentence_2.to(
                device), length_2.to(device), labels.to(device)

        outputs = F.softmax(
            model(sentence_1_batch, length_1_batch, sentence_2_batch, length_2_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, learning_rate, num_epochs, filename):

    print("Number of trainable parameters:{}".format(count_parameters(model)))

    Loss_list = []
    train_acc = []
    val_acc = []
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
#     criterion=criterion.to(device)

    # Train the model
    total_step = len(train_loader)

    for epoch in tqdm.trange(num_epochs):
        for i, (sentence_1, length_1, sentence_2, length_2, labels) in enumerate(train_loader):
            sentence_1_batch, length_1_batch, sentence_2_batch, length_2_batch, labels =\
                sentence_1.to(device), length_1.to(device), sentence_2.to(
                    device), length_2.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            # Forward pass

            outputs = model(sentence_1_batch, length_1_batch,
                            sentence_2_batch, length_2_batch)

            loss = criterion(outputs, labels)
            Loss_list.append(loss)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                train_acc.append(test_model(train_loader, model))
                val_acc.append(test_model(val_loader, model))
                print('Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}, Validation Acc: {}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), Loss_list[-1], val_acc[-1]))

    print("Last model's validation Accuracy:{}".format(val_acc[-1]))
    pk.dump(val_acc, open('./val_acc/' + filename + ".p", 'wb'))
    pk.dump(train_acc, open('./train_acc/' + filename + ".p", 'wb'))
    pk.dump(Loss_list, open('./Loss/' + filename + ".p", 'wb'))

    torch.save(model.state_dict(), "./model_trained/" + filename + ".pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_snli_data_train, processed_snli_data_val, loaded_embeddings_ft = read_data(
    "./data/")

MAX_WORD_LENGTH_1 = int(np.percentile(
    [len(instance[0]) for instance in processed_snli_data_train], 99))
MAX_WORD_LENGTH_2 = int(np.percentile(
    [len(instance[1]) for instance in processed_snli_data_val], 99))

BATCH_SIZE = 64

train_dataset = SNLI_Dataset(processed_snli_data_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=SNLI_collate_func,
                                           shuffle=True)

val_dataset = SNLI_Dataset(processed_snli_data_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=BATCH_SIZE,
                                         collate_fn=SNLI_collate_func,
                                         shuffle=True)


hidden_size_list = [200]
dropout_list = [0]
for hidden_size in hidden_size_list:
    for dropout_rate in dropout_list:
        model = RNN(loaded_embeddings_ft=loaded_embeddings_ft, emb_size=300,
                    hidden_size=hidden_size, drop_out_rate=dropout_rate, num_layers=1, num_classes=3)
        filename = "PY_RNN_Dropout_{}_hidden_size_{}".format(
            dropout_rate, hidden_size)
        train_model(model=model, learning_rate=3e-4,
                    num_epochs=10, filename=filename)

# hidden_size_list = [200, 300, 400]
# dropout_list = [0, 0.5]
# kernel_size_list = [3, 5]
# for hidden_size in hidden_size_list:
#     for dropout_rate in dropout_list:
#         for kernel_size in kernel_size_list:
#             model = CNN(loaded_embeddings_ft=loaded_embeddings_ft, emb_size=300, hidden_size=hidden_size,
#                         dropout_rate=dropout_rate, kernel_size=kernel_size, num_layers=2, num_classes=3)
#             filename = "PY_CNN_Dropout_{}_hidden_size_{}_kernel_size_{}".format(
#                 dropout_rate, hidden_size, kernel_size)
#             train_model(model=model, learning_rate=3e-4,
#                         num_epochs=10, filename=filename)
