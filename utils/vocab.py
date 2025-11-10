import torch
from collections import defaultdict
from torch import nn
from utils.utils import load_vocab
from icecream import ic

# Use to training tokenizer from scratch
class BaseVocab(nn.Module):
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(
            self, 
            vocab_file,
            embedding_dim=768
        ):
        """
            Vocab class to be used when you want to train word embeddings from
            scratch based on a custom vocab. This will initialize the random
            vectors for the vocabulary you pass. Get the vectors using
            `get_vectors` function. This will also create random embeddings for
            some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
            UNK - <unk>.

            Parameters
            ----------
            vocab_file : str
                Path of the vocabulary file containing one word per line
            embedding_dim : int
                Size of the embedding

        """
        super().__init__()
        #-- 1. Init stoi and itos
        self.word_dict = {}
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        self.itos = {}
        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.embedding_dim = embedding_dim
        # ic(vocab_file)
        self.vocabs = load_vocab(vocab_file)
        # ic(self.vocabs)s
        for i, token in enumerate(self.vocabs):
            self.word_dict[token] = 4 + i + 1
            self.itos[4 + i + 1] = token

        self.stoi = defaultdict(lambda: self.UNK_INDEX, self.word_dict)

        #-- 2. Init vector embedding for all word in vocabulary
            #~ Random init matrix: len_vocab x embedding_size
        self.vectors = torch.FloatTensor(self.get_size(), self.embedding_dim)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_pad_index(self):
        return self.PAD_INDEX

    def get_pad_token(self):
        return self.PAD_TOKEN

    def get_start_index(self):
        return self.SOS_INDEX

    def get_start_token(self):
        return self.SOS_TOKEN

    def get_end_index(self):
        return self.EOS_INDEX

    def get_end_token(self):
        return self.EOS_TOKEN

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def get_vectors(self):
        return getattr(self, "vectors", None)
    
    def get_word_idx(self, word):
        return self.stoi[word]
    
    def get_idx_word(self, idx):
        return self.itos[idx]
    
    def get_vector_item(self, idx):
        return self.vectors[idx]

    def get_word_embedding(self, word):
        word_idx = self.get_word_idx(word)
        return self.get_vector_item(word_idx)
        
    

#------------------------------------------------
class PretrainedVocab(BaseVocab):
    def __init__(
            self, 
            model=None,
            tokenizer=None,
            vocab_file=None,
        ):
        """
        Use this vocab class when you have a custom vocabulary class but you
        want to use pretrained embedding vectos for it. This will only load
        the vectors which intersect with your vocabulary. 

        Parameters
        ----------
        vocab_file : str
            Vocabulary file containing list of words with one word per line
            which will be used to collect vectors
        """
        embedding_dim = model.config.hidden_size
        super().__init__(
            vocab_file=vocab_file, 
            embedding_dim=embedding_dim
        )
        self.model = model
        self.tokenizer = tokenizer
        
        # Vector embedding of inforgraphic vocabulary
        self.vectors = nn.Embedding(
            num_embeddings=len(self.stoi),
            embedding_dim=self.embedding_dim
        )
        self.create_embedding()

    
    def create_embedding(self):
        """
        Use to get embedding value of singular word
        """
        vocabs = list(self.stoi.keys())
        # ic(vocabs)

        # Pretrained static features
        embeddings: torch.Tensor = self.model.embeddings.word_embeddings.weight
        vocab_ids = self.tokenizer.convert_tokens_to_ids(
            vocabs
        )
        
        ## DEBUG
        # for word, input_id in zip(vocabs, vocab_ids):
        #     if len(input_id) > 1:
        #         print(f"{word}: {input_id}")
        ## DEBUG

        vocab_embedding = embeddings[vocab_ids]
        self.vectors.weight.data.copy_(vocab_embedding)

    def get_vocab_embedding(self):
        return self.vectors.weight.data

    def get_embedding(self, words):
        """
            Function use to get embed vector of the batch of worf

            Parameters:
            ----------
                words: List[str]
                    List of token

            Output:
            ------
                embed_vector: Tensor: len(words), 768

        """
        word_indices = [self.stoi[word] for word in words]
        return self.vectors(torch.tensor(word_indices))
        

# -------------------------------
class OCRVocabItem:
    def __init__(
            self,
            each_ocr_tokens
        ):
        """
            OCR Vocab for each images
        """
        self.stoi = {token: i for i, token in enumerate(each_ocr_tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}

    def get_word_idx(self, word):
        """
            Get the idx of the word
        """
        return self.stoi[word]
    
    def get_idx_word(self, idx):
        """
            Get the word of the idx
        """
        return self.itos[idx]


class OCRVocab:
    def __init__(
            self,
            ocr_tokens
        ):
        """
        OCR token vocab class to get the matrix features for ocr_token
        
        Parameters:
        ----------
        ocr_tokens: List[List[str]]: BS, num_ocr
            Batch list ocr tokens of images

        Return:
        ----------
            List of OCRVocabItem: List[OCRVocabItem]


        """
        self.batch_size = len(ocr_tokens)

        self.batch_ocr_vocab = []
        for i, tokens in enumerate(ocr_tokens):
            ocr_vocab_item = OCRVocabItem(
                each_ocr_tokens=tokens
            )
            self.batch_ocr_vocab.append(ocr_vocab_item)
        
    def __getitem__(self, idx):
        # Return the OCRVocabItem of corresponding idx 
        return self.batch_ocr_vocab[idx]

    def __len__(self):
        # Get length
        return self.batch_size

