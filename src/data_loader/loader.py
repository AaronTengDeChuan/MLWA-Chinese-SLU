# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/12 19:53
@Author     :       dcteng
@File       :       loader.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/12 19:53
"""

import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer

from src.data_loader.word_trie import Word_Trie

from src.data_loader import functions


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__index2instance:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'w') as fw:
            for element, frequency in self.__counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'w') as fw:
            for index, element in enumerate(self.__index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)

    @property
    def instance2index(self):
        return deepcopy(self.__instance2index)


class NewTorchDataset(Dataset):
    def __init__(self, *inputs):
        self.__inputs = inputs

    def __getitem__(self, index):
        return tuple(field[index] if field is not None else None for field in self.__inputs)

    def __len__(self):
        # Pre-check to avoid bug.
        length = set(len(field) for field in self.__inputs if field is not None)
        assert len(length) == 1
        return length.pop()


class DatasetManager(object):
    # TODO: divide original text into char-level text and word-level text, and provide alignment information between two level texts

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        if args.unique_vocabulary:
            self.__char_alphabet = self.__word_alphabet
        else:
            self.__char_alphabet = Alphabet('char', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        self.__char_embedding = None
        self.__word_embedding = None

        self.__max_char_seq_len = 50

        if args.use_bert_input:
            self.__bert_tokenizer = BertTokenizer.from_pretrained(functions.MODEL_PATH_MAP["chinese_bert"])
            self.__bert_input_id_data = {}
            self.__bert_attention_mask_data = {}
            self.__bert_token_type_data = {}

        if args.use_lexicon_gnn or args.use_simple_lexicon_gnn:
            self.number_normalized = True
            self.__word_dict = Word_Trie()
            self.__text_word_list_data = {}
            self.__digit_word_list_data = {}
            self.__digit_sent_seg_data = {}

        # Record the raw text of dataset.
        self.__text_char_data = {}
        self.__text_word_data = {}
        self.__text_align_info = {}
        self.__text_slot_data = {}
        self.__text_intent_data = {}

        # Record the serialization of dataset.
        self.__digit_char_data = {}
        self.__digit_word_data = {}
        self.__digit_align_info = {}
        self.__digit_slot_data = {}
        self.__digit_intent_data = {}

        self.__args = args

    @property
    def test_sentence(self):
        return deepcopy(self.__text_char_data['test'])

    @property
    def char_alphabet(self):
        return deepcopy(self.__char_alphabet)

    @property
    def word_alphabet(self):
        return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def intent_alphabet(self):
        return deepcopy(self.__intent_alphabet)

    @property
    def char_embedding(self):
        return deepcopy(self.__char_embedding)

    @property
    def word_embedding(self):
        return deepcopy(self.__word_embedding)

    @property
    def num_training_samples(self):
        return len(self.__text_char_data['train'])

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def max_grad_norm(self):
        return self.__args.max_grad_norm

    @property
    def save_dir(self):
        return self.__args.save_dir

    # @property
    # def intent_forcing_rate(self):
    #     return self.__args.intent_forcing_rate

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate

    @property
    def unique_vocabulary(self):
        return self.__args.unique_vocabulary

    @property
    def golden_intent(self):
        return self.__args.golden_intent

    @property
    def no_progressbar(self):
        return self.__args.no_progressbar

    @property
    def use_bert_input(self):
        return self.__args.use_bert_input

    # @property
    # def token_forcing_rate(self):
    #     return self.__args.token_forcing_rate

    # @property
    # def tokenization(self):
    #     return self.__args.tokenization

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        print('\tnumber of train sample:                    {};'.format(len(self.__text_char_data['train'])))
        print('\tnumber of dev sample:                      {};'.format(len(self.__text_char_data['dev'])))
        print('\tnumber of test sample:                     {};'.format(len(self.__text_char_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))
        print('\tteacher forcing rate(slot)		    		{};'.format(self.slot_forcing_rate))
        print('\tunique vocabulary:                         {};'.format(self.unique_vocabulary))
        print('\tgolden intent:                             {};\n'.format(self.golden_intent))

        print('\tuse bert input:                            {};\n'.format(self.use_bert_input))

        print('\tuse lexicon gnn:                           {};'.format(self.__args.use_lexicon_gnn))
        print('\tallow single char:                         {};'.format(self.__args.allow_single_char))
        print('\tuse simple lexicon gnn:                    {};'.format(self.__args.use_simple_lexicon_gnn))
        print('\tuse pretrained embeddings:                 {};'.format(self.__args.use_pretrained_emb))
        print('\tFind all sentence segmentation combination:{};'.format(self.__args.full_sent_seg))
        print('\tchar embedding path:                       {};'.format(self.__args.char_emb_path))
        print('\tword embedding path:                       {};\n'.format(self.__args.word_emb_path))
        # print('\tteacher forcing rate(intent):		    	{};'.format(self.intent_forcing_rate))
        # print('\tteacher forcing rate(token):		    	{};'.format(self.token_forcing_rate))
        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    def save_align_info(self, dir_path, data_name):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        if self.__args.use_lexicon_gnn or self.__args.use_simple_lexicon_gnn:
            word_list_data = self.__text_word_list_data[data_name]
        else:
            word_list_data = [[[] for _ in range(len(char_text))] for char_text in self.__text_char_data[data_name]]

        if self.__args.use_lexicon_gnn:
            sent_seg_data = self.__digit_sent_seg_data[data_name]
        else:
            sent_seg_data =[[] for _ in range(len(self.__text_char_data[data_name]))]

        dict_path = os.path.join(dir_path, "{}_alignment_info.txt".format(data_name))
        with open(dict_path, 'w') as fw:
            for char_text, word_text, align_info, word_list, sent_seg in zip(
                    self.__text_char_data[data_name],
                    self.__text_word_data[data_name],
                    self.__text_align_info[data_name],
                    word_list_data, sent_seg_data):
                start = 0
                for i, align in enumerate(align_info):
                    for j in range(start, start + align):
                        fw.write("{}\t{}\t{}\t{}\n".format(char_text[j], word_text[i], i, " ".join(word_list[j])))
                    start += align
                for seg in sent_seg:
                    fw.write("{}:\t{}\t{}\n".format(seg[5], " ".join(seg[0]), str(seg[4])))
                fw.write("\n")

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """
        # read word lexicon
        if self.__args.use_lexicon_gnn or self.__args.use_simple_lexicon_gnn:
            self.build_word_file(self.__args.word_emb_path)

        # read data files
        train_path = os.path.join(self.__args.data_dir, self.__args.train_file_name)
        dev_path = os.path.join(self.__args.data_dir, self.__args.valid_file_name)
        test_path = os.path.join(self.__args.data_dir, self.__args.test_file_name)

        self.add_file(train_path, 'train', if_train_file=True)
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        # read embedding files
        if self.__args.use_pretrained_emb:
            self.__word_embedding, self.__args.word_embedding_dim = functions.build_pretrain_embedding(
                self.__args.word_emb_path, self.__word_alphabet, norm=True)

            self.__char_embedding, self.__args.char_embedding_dim = functions.build_pretrain_embedding(
                self.__args.char_emb_path, self.__char_alphabet, norm=True,
                pre_embedding=self.__word_embedding if self.__args.unique_vocabulary else None)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.__args.save_dir, "alphabet")
        self.__char_alphabet.save_content(alphabet_dir)
        self.__word_alphabet.save_content(alphabet_dir)
        self.save_align_info(alphabet_dir, "train")
        self.save_align_info(alphabet_dir, "dev")
        self.save_align_info(alphabet_dir, "test")
        self.__slot_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)

    def get_dataset(self, data_name, is_digital):
        """ Get dataset of given unique name.

        :param data_name: is name of stored dataset.
        :param is_digital: make sure if want serialized data.
        :return: the required dataset.
        """

        if is_digital:
            return self.__digit_char_data[data_name], \
                   self.__digit_word_data[data_name], \
                   self.__digit_align_info[data_name], \
                   self.__digit_slot_data[data_name], \
                   self.__digit_intent_data[data_name]
        else:
            return self.__text_char_data[data_name], \
                   self.__text_word_data[data_name], \
                   self.__text_align_info[data_name], \
                   self.__text_slot_data[data_name], \
                   self.__text_intent_data[data_name]

    def build_word_file(self, word_file):
        # build word file,initial word embedding file
        with open(word_file, 'r', encoding="utf-8") as f:
            for line in f:
                word = line.strip().split()[0]
                if word:
                    self.__word_dict.insert(word)
        print("Building the word dict...")

    def add_file(self, file_path, data_name, if_train_file):
        char_text, word_text, align_info, slot, intent = \
            self.__read_file(file_path, self.__max_char_seq_len - (2 if self.__args.use_bert_input else 0))

        if if_train_file:
            self.__char_alphabet.add_instance(char_text)
            self.__word_alphabet.add_instance(word_text)
            self.__slot_alphabet.add_instance(slot)
            self.__intent_alphabet.add_instance(intent)

        if self.__args.use_bert_input:
            input_ids, attention_mask, token_type_ids = \
                functions.convert_examples_to_features(char_text, self.__max_char_seq_len, self.__bert_tokenizer)
            self.__bert_input_id_data[data_name] = input_ids
            self.__bert_attention_mask_data[data_name] = attention_mask
            self.__bert_token_type_data[data_name] = token_type_ids

        if self.__args.use_lexicon_gnn or self.__args.use_simple_lexicon_gnn:
            if not self.__args.use_simple_lexicon_gnn: self.__word_dict.insert(word_text)
            instance_text = [
                self.__read_instance_with_gaz(chars, self.__word_dict, word_alphabet=None, allow_single_char=self.__args.allow_single_char) for chars in char_text]
            self.__word_alphabet.add_instance(instance_text)
            self.__text_word_list_data[data_name] = instance_text
            self.__digit_word_list_data[data_name] = [
                self.__read_instance_with_gaz(instance, None, word_alphabet=self.__word_alphabet) for instance in instance_text]
            if not self.__args.use_simple_lexicon_gnn:
                self.__digit_sent_seg_data[data_name] = \
                    functions.build_sentence_segmentation(word_text, instance_text, self.no_progressbar, self.__args.full_sent_seg)

        # Record the raw text of dataset.
        self.__text_char_data[data_name] = char_text
        self.__text_word_data[data_name] = word_text
        self.__text_align_info[data_name] = align_info
        self.__text_slot_data[data_name] = slot
        self.__text_intent_data[data_name] = intent

        # Serialize raw text and stored it.
        self.__digit_char_data[data_name] = self.__char_alphabet.get_index(char_text)
        self.__digit_word_data[data_name] = self.__word_alphabet.get_index(word_text)
        self.__digit_align_info[data_name] = align_info
        if if_train_file:
            self.__digit_slot_data[data_name] = self.__slot_alphabet.get_index(slot)
            self.__digit_intent_data[data_name] = self.__intent_alphabet.get_index(intent)

    @staticmethod
    def __read_instance_with_gaz(tokens, word_dict, word_alphabet=None, allow_single_char=False):
        words = []
        word_Ids = []
        if word_alphabet is None:
            for idx in range(len(tokens)):
                matched_list = word_dict.recursive_search(tokens[idx:], allow_single_char=allow_single_char)
                words.append(matched_list)
            return words
        else:
            for matched_list in tokens:
                matched_length = [len(a) for a in matched_list]
                matched_Id = [word_alphabet.get_index(word) for word in matched_list]
                if matched_Id:
                    word_Ids.append([matched_Id, matched_length])
                else:
                    word_Ids.append([])
            return word_Ids


    @staticmethod
    def __read_file(file_path, max_char_seq_len):
        """ Read data file of given path.

        :param file_path: path of data file.
        :return: list of sentence (chars), list of sentence (words), list of align info, list of slot and list of intent.
        """

        def endOfChunk(pre_tag, tag):
            if pre_tag == "B" and tag == "B":
                return True
            if pre_tag == "E" and tag == "B" or pre_tag == "E" and tag == "S":
                return True
            if pre_tag == "S" and tag == "S" or pre_tag == "S" and tag == "B":
                return True
            return False

        char_texts, word_texts, align_infos, slots, intents = [], [], [], [], []

        char_text, word_text, align_info, slot = [], [], [], []
        pre_tag, word = "S", ""

        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    if len(word) != 0:
                        word_text.append(word)
                        align_info.append(len(word))
                    assert len(char_text) == sum(align_info)
                    char_texts.append(char_text)
                    word_texts.append(word_text)
                    align_infos.append(align_info)
                    slots.append(slot)
                    intents.append(items)

                    # clear buffer lists.
                    char_text, word_text, align_info, slot = [], [], [], []
                    pre_tag, word = "S", ""

                elif len(items) >= 2:
                    char, slot_tag = items[0].strip(), items[1].strip()
                    if len(char_text) >= max_char_seq_len:
                        continue
                    char_text.append(char)
                    slot.append(slot_tag)
                    if len(items) == 2:
                        word_text.append(char)
                    elif len(items) == 3:
                        tag = items[2].strip()
                        if endOfChunk(pre_tag, tag):
                            if len(word) != 0:
                                word_text.append(word)
                                align_info.append(len(word))
                            word = char
                        else:
                            word += char
                        pre_tag = tag

        return char_texts, word_texts, align_infos, slots, intents

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            char_text = self.__digit_char_data[data_name]
            word_text = self.__digit_word_data[data_name]
            align_info = self.__digit_align_info[data_name]
            slot = self.__digit_slot_data[data_name]
            intent = self.__digit_intent_data[data_name]
        else:
            char_text = self.__text_char_data[data_name]
            word_text = self.__text_word_data[data_name]
            align_info = self.__text_align_info[data_name]
            slot = self.__text_slot_data[data_name]
            intent = self.__text_intent_data[data_name]

        word_list = self.__digit_word_list_data[data_name] \
            if self.__args.use_lexicon_gnn or self.__args.use_simple_lexicon_gnn else None
        sent_seg = self.__digit_sent_seg_data[data_name] if self.__args.use_lexicon_gnn else None
        bert_input_id = self.__bert_input_id_data[data_name] if self.__args.use_bert_input else None
        bert_attention_mask = self.__bert_attention_mask_data[data_name] if self.__args.use_bert_input else None
        bert_token_type = self.__bert_token_type_data[data_name] if self.__args.use_bert_input else None

        dataset = NewTorchDataset(char_text, word_text, align_info, slot, intent, word_list, sent_seg,
                                  bert_input_id, bert_attention_mask, bert_token_type)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    @staticmethod
    def add_padding(char_texts, word_texts, char_items=None, word_items=None, digital=True):
        char_len_list = [len(text) for text in char_texts]
        char_max_len = max(char_len_list)
        word_len_list = [len(text) for text in word_texts]
        word_max_len = max(word_len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(char_len_list)[::-1]

        trans_char_texts, char_seq_lens, trans_char_items = [], [], None
        trans_word_texts, word_seq_lens, trans_word_items = [], [], None
        if char_items is not None:
            trans_char_items = [[] for _ in range(0, len(char_items))]
        if word_items is not None:
            trans_word_items = [[] for _ in range(0, len(word_items))]

        for index in sorted_index:
            char_seq_lens.append(deepcopy(char_len_list[index]))
            word_seq_lens.append(deepcopy(word_len_list[index]))
            trans_char_texts.append(deepcopy(char_texts[index]))
            trans_word_texts.append(deepcopy(word_texts[index]))
            if digital:
                trans_char_texts[-1].extend([0] * (char_max_len - char_len_list[index]))
                trans_word_texts[-1].extend([0] * (word_max_len - word_len_list[index]))
            else:
                trans_char_texts[-1].extend(['<PAD>'] * (char_max_len - char_len_list[index]))
                trans_word_texts[-1].extend(['<PAD>'] * (word_max_len - word_len_list[index]))

            # This required specific if padding after sorting.
            if char_items is not None:
                for item, (o_item, required) in zip(trans_char_items, char_items):
                    item.append(deepcopy(o_item[index]) if o_item else None)
                    if required:
                        if digital:
                            item[-1].extend([0] * (char_max_len - char_len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (char_max_len - char_len_list[index]))

            if word_items is not None:
                for item, (o_item, required) in zip(trans_word_items, word_items):
                    item.append(deepcopy(o_item[index]) if o_item else None)
                    if required:
                        if digital:
                            item[-1].extend([0] * (word_max_len - word_len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (word_max_len - word_len_list[index]))

        if char_items is not None and word_items is not None:
            return trans_char_texts, trans_word_texts, trans_char_items, trans_word_items, char_seq_lens, word_seq_lens, sorted_index
        elif char_items is not None:
            return trans_char_texts, trans_word_texts, trans_char_items, char_seq_lens, word_seq_lens, sorted_index
        elif word_items is not None:
            return trans_char_texts, trans_word_texts, trans_word_items, char_seq_lens, word_seq_lens, sorted_index
        else:
            return trans_char_texts, trans_word_texts, char_seq_lens, word_seq_lens, sorted_index

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch