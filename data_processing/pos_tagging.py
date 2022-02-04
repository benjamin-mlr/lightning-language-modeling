

import datasets
import numpy as np

from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,

)


class POSUDDataModule(LightningDataModule):
    """ Data loading and processing class for UD POS tagging for pytorch lightning training """


    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
        "is_token_bo_word",
        #"upos",
        #"sentence_length_gt_max_len",
        "sentence_length_in_words",
    ]
    DATASET_NAME = "universal_dependencies"
    TEXT_FIELD = "tokens"
    LABEL_FIELDS = ["upos"]
    NUM_LABELS = 17
    LABEL_TO_IGNORE = -1

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        treebank="en_ewt",
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.treebank = treebank
        self.label_to_ignore = self.LABEL_TO_IGNORE

        self.text_field = self.TEXT_FIELD
        self.num_labels = self.NUM_LABELS
        self.label_fields = self.LABEL_FIELDS

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)


    def setup(self, stage: str):
        self.dataset = datasets.load_dataset(self.DATASET_NAME, self.treebank)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=self.label_fields)

            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset(self.DATASET_NAME, self.treebank)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def remove_mwe_token(self, tokens, labels, label_to_remove=13):
        """
        In the datasets library, mwe (UD tokens that have INDEX_1-INDEX_2 as indexes
        are included in the tokens list: we need to remove them to do word level sequence labelling
        :return:
        """
        assert len(tokens) == len(labels)
        index_to_remove = [[] for _ in range(len(tokens))]

        for ind_sample, seq_label in enumerate(labels):
            _index_to_remove = index_to_remove[ind_sample]

            for i, label in enumerate(seq_label):
                if label == label_to_remove:
                    _index_to_remove.append(i)

            assert len(tokens[ind_sample]) == len(labels[ind_sample]), "pre mwe removal: unaligned"
            former_len = len(tokens[ind_sample])
            if len(_index_to_remove):
                tokens[ind_sample] = np.delete(tokens[ind_sample], _index_to_remove).tolist()
                labels[ind_sample] = np.delete(labels[ind_sample], _index_to_remove).tolist()
            assert len(tokens[ind_sample]) == len(labels[ind_sample]), "post mwe removal: unaligned"
            assert len(tokens[ind_sample])+len(_index_to_remove) == former_len

    def get_aligned_label_with_bpe(self, input_features, labels):
        """
        Given labels (aligned with original UD words) and bpe sequence
        - create variable with bpe-aligned labels: each non-starting bpe (of words) get -1 as label OR special tokens get -1
        :param input_features:
        :param labels:
        :return:
        """
        label_aligned = [[] for _ in labels]
        is_token_bo_word = [[] for _ in labels]

        for ind_batch, len_seq in enumerate(input_features.length):
            word_id_prev = None
            for ind_token in range(len_seq):
                word_id = input_features.token_to_word(ind_batch, ind_token)

                if word_id is None or word_id == word_id_prev:
                    # word_id is None refers to BOS/EOS/PAD tokens
                    label_aligned[ind_batch].append(self.label_to_ignore)
                    is_token_bo_word[ind_batch].append(0)
                else:
                    label_aligned[ind_batch].append(labels[ind_batch][word_id])
                    is_token_bo_word[ind_batch].append(1)

                word_id_prev = word_id
            assert len_seq == len(label_aligned[ind_batch])
        return label_aligned, is_token_bo_word

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs

        texts_or_text_pairs = example_batch[self.text_field]

        # remove MWE tokens and associated labels
        if len(self.label_fields) == 1:
            self.remove_mwe_token(texts_or_text_pairs, example_batch[self.label_fields[0]])

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length,
            pad_to_max_length=True,
            return_length=True,
            truncation=True, is_split_into_words=True
        )
        features["sentence_length_in_words"] = [len(sent) for sent in texts_or_text_pairs]
        features["sentence_length_in_bpe"] = [len(sent) for sent in texts_or_text_pairs]
        # if last id equals eos_token_id it means we are (most likely) truncating tokens
        features["sentence_length_gt_max_len"] = [input_id[-1] == self.tokenizer.eos_token_id for input_id in features["input_ids"]]
        if sum(features["sentence_length_gt_max_len"]) > 0:
            print(f'Warning: {sum(features["sentence_length_gt_max_len"])}/{len(features["sentence_length_gt_max_len"])} sentences longer than max length {self.max_seq_length}')

        if len(self.label_fields) == 1:
            features[self.label_fields[0]] = example_batch[self.label_fields[0]]
            features["labels"], features["is_token_bo_word"] = self.get_aligned_label_with_bpe(features, example_batch[self.label_fields[0]])
        else:
            raise(Exception(f"{self.label_fields} only label output set supported"))

        return features

    def sanity_test_alignement(self):
        """
        sanity testing that :
        - bpe and labels are aligned
        - bpe that are not begining of word (do not start with _") are associated with -1 label
        - words and original labels are aligned
        :return:
        """

        iterator = self.train_dataloader()

        BEGINING_OF_WORD_CHAR = "▁"
        print(f"\nWarning: for sanity checking we assume that after tokenization, all UD words should start with the symbol '{BEGINING_OF_WORD_CHAR}'")

        for batch in tqdm(iterator, "Sanity checking alignement words, bpes, labels is ok"):

            # Sanity check sentence length with label length
            for _upos, _len in zip(batch["upos"], batch["sentence_length_in_words"]):
                assert len(_upos) == _len

            # Sanity check sentence length with label length
            for _input_ids, _labels in zip(batch["input_ids"], batch["labels"]):
                # padded sequences should have the same length
                assert len(_input_ids) == len(_labels)
                # unpadded sequences should have the same length after removing self.label_to_ignore:
                #   cannot sanity check because label -1 is both for padded symbol than non-first token symbols

                # Sanity check that bpe that are not begining of word (do not start with _") are associated with -1 label
                for id, label in zip(_input_ids, _labels):
                    token = self.tokenizer.convert_ids_to_tokens(id.item())
                    if not token.startswith(BEGINING_OF_WORD_CHAR) or token in [self.tokenizer.pad_token, self.tokenizer.bos_token, self.tokenizer.eos_token]:
                        assert label.item() == self.label_to_ignore



if __name__ == "__main__":

    from datasets import load_dataset
    dataset = load_dataset("universal_dependencies", "ar_padt", "validation")
    breakpoint()

    dm = POSUDDataModule("xlm-roberta-base", max_seq_length=256, train_batch_size=9, treebank="fr_sequoia")
    # MWE: are annotated with 13 UDPOS : correct ??
    dm.prepare_data()
    dm.setup("fit")

    # SANITY CHECKING REQUIRES TO KEEP UNPADDED LABEL--> requires batch 1
    #dm.sanity_test_alignement()

    _iter = dm.train_dataloader()
    for i, a in enumerate(_iter):
        print("INDEX ----- ", i)

        print(a["input_ids"].size(), a["attention_mask"].size(), a["labels"].size())






