import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNN(nn.Module):
    def __init__(self, loaded_embeddings_ft, emb_size, hidden_size, drop_out_rate, num_layers, num_classes):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size, self.drop_out_rate = num_layers, hidden_size, drop_out_rate
        #print(self.hidden_size_1, self.hidden_size_2, self.hidden_size_3)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(loaded_embeddings_ft)).float().to(device)
        self.rnn_1 = nn.GRU(emb_size, self.hidden_size, self.num_layers, batch_first=True,  bidirectional=True).to(device)
        self.rnn_2 = nn.GRU(emb_size, self.hidden_size, self.num_layers, batch_first=True,  bidirectional=True).to(device)
        self.relu = nn.ReLU().to(device)
        self.linear_1 = nn.Linear(self.hidden_size+self.hidden_size, hidden_size).to(device)
        self.linear_2 = nn.Linear(self.hidden_size, num_classes).to(device)
        self.dropout = nn.Dropout(self.drop_out_rate).to(device)

    def init_hidden(self, batch_size, hidden_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers*2, batch_size, hidden_size)

        return hidden.to(device)

    def forward(self, sentence_1, length_1, sentence_2, length_2):
        
        sentence_1=sentence_1
        length_1=length_1
        sentence_2=sentence_2
        length_2=length_2

        batch_size, seq_len_1 = sentence_1.size()
        batch_size, seq_len_2 = sentence_2.size()
        
        length_1_sorted, idx_1_sorted = torch.sort(length_1, dim=0, descending=True)
        length_2_sorted, idx_2_sorted = torch.sort(length_2, dim=0, descending=True)
        
        _, idx_1_unsorted = torch.sort(idx_1_sorted, dim=0)
        _, idx_2_unsorted = torch.sort(idx_2_sorted, dim=0)
        
        length_1=length_1.index_select(0, idx_1_sorted)
        length_2=length_2.index_select(0, idx_2_sorted)
        
        # reset hidden state
        self.hidden_1 = self.init_hidden(batch_size, self.hidden_size)
        self.hidden_2 = self.init_hidden(batch_size, self.hidden_size)
        
#         print (self.hidden_1.shape)
#         print (self.hidden_2.shape)
        

        # get embedding of characters
        embed_1 = self.embedding(sentence_1)
        embed_2 = self.embedding(sentence_2)
        
        # sort embed from max length to min length 
        embed_1 = embed_1.index_select(0, idx_1_sorted)
        embed_2 = embed_2.index_select(0, idx_2_sorted)
        
        # pack padded sequence
        embed_1 = torch.nn.utils.rnn.pack_padded_sequence(embed_1, length_1.cpu().numpy(), batch_first=True)
        embed_2 = torch.nn.utils.rnn.pack_padded_sequence(embed_2, length_2.cpu().numpy(), batch_first=True)
        
#         print (type(embed_1.cuda()))
#         print (type(self.hidden_1.cuda()))
    
        # fprop though RNN
        rnn_out_1, self.hidden_1= self.rnn_1(embed_1, self.hidden_1)#.to(device)
        rnn_out_2, self.hidden_2= self.rnn_2(embed_2, self.hidden_2)#.to(device)

        # combine vector

        #concatenated_vector =torch.cat ([self.hidden1, self.hidden2],dim=-1)
        
        # undo packing
#         rnn_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_1, batch_first=True)
#         rnn_out_2, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_2, batch_first=True)
        
       
        
        ##
        self.hidden_1 = self.hidden_1.index_select(1, idx_1_unsorted)
        self.hidden_2 = self.hidden_2.index_select(1, idx_2_unsorted)

#         print (self.hidden_1.shape)
#         print (self.hidden_2.shape)
        # combine vector
        concatenated_vector =torch.cat ([self.hidden_1, self.hidden_2],dim=2)
        
        rnn_output = torch.sum(concatenated_vector, dim=0)#.to(device)
#         print (rnn_output.shape)
        rnn_output = self.linear_1(rnn_output)#.to(device)
        
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.relu(rnn_output)#.to(device)
        
        logits = self.linear_2(rnn_output)#.to(device)
        
        return logits#.to(device)