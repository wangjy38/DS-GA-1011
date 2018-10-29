import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_WORD_LENGTH_1=34
MAX_WORD_LENGTH_2=19


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, loaded_embeddings_ft, emb_size, hidden_size, dropout_rate, kernel_size, num_layers, num_classes):

        super(CNN, self).__init__()
        
        padding=(kernel_size-1)//2
        
        self.num_layers, self.hidden_size, self.dropout_rate = num_layers, hidden_size, dropout_rate
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(loaded_embeddings_ft)).float().to(device)
    
        self.conv1 = nn.Conv1d(emb_size, self.hidden_size, kernel_size=kernel_size, padding=padding).to(device)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=kernel_size, padding=padding).to(device)

        self.conv3 = nn.Conv1d(emb_size, self.hidden_size, kernel_size=kernel_size, padding=padding).to(device)
        self.conv4 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=kernel_size, padding=padding).to(device)
        
        self.maxpool_1 = nn.MaxPool1d(MAX_WORD_LENGTH_1).to(device)
        self.maxpool_2 = nn.MaxPool1d(MAX_WORD_LENGTH_2).to(device)
        
        self.linear_1 = nn.Linear(2*hidden_size, self.hidden_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.linear_2 = nn.Linear(self.hidden_size, num_classes).to(device)
        self.dropout = nn.Dropout(self.dropout_rate).to(device)

    def forward(self, sentence_1, length_1, sentence_2, length_2):
        
        batch_size_1, seq_len_1 = sentence_1.size()
        batch_size_2, seq_len_2 = sentence_2.size()
        
        embed_1 = self.embedding(sentence_1)
        embed_2 = self.embedding(sentence_2)
        
        #print (embed_1.shape, embed_2.shape)
        
        hidden_1 = self.conv1(embed_1.transpose(1,2)).transpose(1,2)
#         print (hidden_1.shape)
#         print (hidden_1.contiguous().view(-1, hidden_1.size(-1)).shape)
        
        hidden_1 = F.relu(hidden_1.contiguous().view(-1, hidden_1.size(-1))).view(batch_size_1, seq_len_1, hidden_1.size(-1))
        
        #print (hidden_1.shape)
        
        hidden_1 = self.conv2(hidden_1.transpose(1,2)).transpose(1,2)
        hidden_1 = F.relu(hidden_1.contiguous().view(-1, hidden_1.size(-1))).view(batch_size_1,  hidden_1.size(-1), seq_len_1)

        hidden_1 = self.maxpool_1(hidden_1)
        
        hidden_2 = self.conv1(embed_2.transpose(1,2)).transpose(1,2)
        hidden_2 = F.relu(hidden_2.contiguous().view(-1, hidden_2.size(-1))).view(batch_size_2, seq_len_2, hidden_2.size(-1))

        hidden_2 = self.conv2(hidden_2.transpose(1,2)).transpose(1,2)
        
        hidden_2 = F.relu(hidden_2.contiguous().view(-1, hidden_2.size(-1))).view(batch_size_2,  hidden_2.size(-1), seq_len_2)
        #print (hidden_2.shape)
        hidden_2 = self.maxpool_2(hidden_2)
        
        #print (hidden_1.shape, hidden_2.shape)
        concatenated_vector =torch.cat ([hidden_1, hidden_2],dim=1)
        
        #print (concatenated_vector.shape)
        
        cnn_output = torch.sum(concatenated_vector, dim=2)
        
        #print (cnn_output.shape)
        
        cnn_output= self.linear_1(cnn_output)
        cnn_output = self.dropout(cnn_output)
        cnn_output= self.relu(cnn_output)
        
        logits= self.linear_2(cnn_output)

        return logits#.to(device)