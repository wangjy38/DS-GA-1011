from torch.utils.data import Dataset
import numpy as np
import torch
MAX_WORD_LENGTH_1 = 34
MAX_WORD_LENGTH_2 = 19
class SNLI_Dataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_tuple):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.data_list_1, self.data_list_2, self.target_list = zip(*data_tuple)
        assert (len(self.data_list_1) == len(self.target_list))
        assert (len(self.data_list_2) == len(self.target_list))
        #self.char2id = char2id

    def __len__(self):
        return len(self.data_list_1)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        sentence_1 =  self.data_list_1[key][:MAX_WORD_LENGTH_1]
        sentence_2 =  self.data_list_2[key][:MAX_WORD_LENGTH_2]
        label = self.target_list[key]
        return [sentence_1, len(sentence_1), sentence_2, len(sentence_2), label]

def SNLI_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list_1 = []
    data_list_2 = []
    label_list = []
    length_list_1 = []
    length_list_2 = []

    for datum in batch:
        label_list.append(datum[4])
        length_list_1.append(datum[1])
        length_list_2.append(datum[3])
    # padding
    for datum in batch:
        #print (MAX_WORD_LENGTH_1-datum[1])
        padded_vec_1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_WORD_LENGTH_1-datum[1])),
                                mode="constant", constant_values=0)
        
        padded_vec_2 = np.pad(np.array(datum[2]),
                                pad_width=((0,MAX_WORD_LENGTH_2-datum[3])),
                                mode="constant", constant_values=0)
        
        data_list_1.append(padded_vec_1)
        
        data_list_2.append(padded_vec_2)
        
#     ind_dec_order = np.argsort(length_list_)[::-1]
#     data_list = np.array(data_list)[ind_dec_order]
#     length_list = np.array(length_list)[ind_dec_order]
#     label_list = np.array(label_list)[ind_dec_order]
    
#     print (type(torch.from_numpy(np.array(data_list_1))))
#     print (type(torch.LongTensor(length_list_1)))
#     print (data_list_2)
#     print (type(torch.from_numpy(np.array(data_list_2))))
#     print (type(torch.LongTensor(length_list_2)))
#     print (type(torch.LongTensor(label_list)))
    
    return [torch.from_numpy(np.array(data_list_1)), torch.LongTensor(length_list_1), torch.from_numpy(np.array(data_list_2)), torch.LongTensor(length_list_2), torch.LongTensor(label_list)]

