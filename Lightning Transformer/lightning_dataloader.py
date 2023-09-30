from pytorch_lightning import LightningDataModule
from config import get_config
from datasets import load_dataset
from dataset import BilingualDataset
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
import torch


myconfig = get_config()

class OpusLightningDataset(LightningDataModule):
    def __init__(self, config = myconfig):
        super().__init__()
        self.config = config
        self.train_data = None
        self.val_data = None

        self.tokenizer_src = None
        self.tokenizer_tgt = None
    
    def prepare_data(self):
        load_dataset('opus_books',
                    f"{self.config['lang_src']}-{self.config['lang_tgt']}",
                     split='train')
    
    def setup(self, stage = None):
        if not self.train_data and not self.val_data:
            ds_raw = load_dataset('opus_books', f"{self.config['lang_src']}-{self.config['lang_tgt']}", split='train')

            # Build tokenizers
            self.tokenizer_src = self.get_or_build_tokenizer(self.config, ds_raw, self.config['lang_src'])
            self.tokenizer_tgt = self.get_or_build_tokenizer(self.config, ds_raw, self.config['lang_tgt'])

            # Keep 90% for training, 10% for validation
            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

            self.train_data = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len'])
            self.val_data = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len'])

            # Find the maximum length of each sentence in the source and target sentence
            max_len_src = 0
            max_len_tgt = 0

            for item in ds_raw:
                src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
                tgt_ids = self.tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))

            print(f'Max length of source sentence: {max_len_src}')
            print(f'Max length of target sentence: {max_len_tgt}')

            print(f"Source Tokenizer Vocab Size : {self.tokenizer_src.get_vocab_size()}")
            print(f"Target Tokenizer Vocab Size : {self.tokenizer_tgt.get_vocab_size()}")
            print("\n")
            
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=True)
            
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=True)

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]

    def get_or_build_tokenizer(self, config, ds, lang):
        tokenizer_path = Path(config["tokenizer_file"].format(lang))

        if not Path.exists(tokenizer_path):
            # code inspired from huggingface tokenizers
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    

def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it or every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=0,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)