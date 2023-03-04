import pickle
import torch
from torchtext.data.utils import get_tokenizer

class Translation(object):
    """
    Translation.
    """
    def __init__(self, src_path, tgt_path, device):
        self.vocab_src = self.read_vocab(src_path)
        self.vocab_tgt = self.read_vocab(tgt_path)
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.token_transform = get_tokenizer('basic_english')
        self.text_transform = self.sequential_transforms(self.token_transform, #Tokenization
                                                self.vocab_src, #Numericalization
                                                self.tensor_transform) # Add BOS/EOS and create tensor
        self.DEVICE = device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # function to generate output sequence using greedy algorithm
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.to(self.DEVICE)
        src_mask = src_mask.to(self.DEVICE)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.DEVICE)
        for i in range(max_len-1):
            memory = memory.to(self.DEVICE)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys


    # actual function to translate input sentence into target language
    def translate(self, model, src_sentence):
        model.eval()
        src = self.text_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=self.BOS_IDX).flatten()
        return " ".join(self.vocab_tgt.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")  

    def len_vocab(self):
        return len(self.vocab_src), len(self.vocab_tgt)

    def read_vocab(self, path):
        #import pickle
        pkl_file = open(path, 'rb')
        vocab = pickle.load(pkl_file)
        pkl_file.close()
        return vocab

    # helper function to club together sequential operations
    def sequential_transforms(self,*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                        torch.tensor(token_ids),
                        torch.tensor([self.EOS_IDX])))