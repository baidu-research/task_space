import numpy as np
from functools import reduce


def create_batch(sentences, tokenizer, bpe_tok=False, return_sentence=False):
    """
        Given a list of sentences, returns:
        1. a batch of token ids as input to a (huggingface) model
        2. the index of first subword for each word (list of list)
    """
    special_tokens = [tokenizer.convert_tokens_to_ids(_) for _ in \
                        tokenizer.special_tokens_map.values()]
    # exclude <unk>
    special_tokens = [_ for _ in special_tokens if _ != tokenizer.unk_token_id]

    # get indice of the token that corresponds to the first word in the sentence
    def sentence_starts_from(token_id_list):    
        for i, t in enumerate(token_id_list):
            if t not in special_tokens:
                return i

    # Huggingface LMs have different tokenizers.
    # The following two tokenization approaches may not have the same results
    # 1) tokenize the whole sentence all at once (default)
    # 2) tokenize word by word, then concatenate
    # 1) is well documented for feature extraction. But we also need 2) to
    # find out word boundary. This function checks if 1) and 2) give the same results
    # If not, error msg is raised
    def word_boundary_sanity(sentence):
        whole_tokenize = tokenizer(sentence.split(),
                                   add_special_tokens=False,
                                   is_split_into_words=True)
        whole_list = tokenizer.convert_ids_to_tokens(whole_tokenize['input_ids'])
        
        if bpe_tok:
            # for BPE encoders used in GPT2, roberta, etc
            wordwise_list = [tokenizer.tokenize(' '+w) for w in sentence.split()]
            subword_lens = [len(_) for _ in wordwise_list]
            wordwise_list = reduce(lambda a, b: a+b, wordwise_list)
            assert whole_list == wordwise_list
        else:
            wordwise_list = [tokenizer.tokenize(w) for w in sentence.split()]
            subword_lens = [len(_) for _ in wordwise_list]
            wordwise_list = reduce(lambda a, b: a+b, wordwise_list)
            assert len(whole_list) == len(wordwise_list)
        return subword_lens

    # call tokenizer on "pre-tokenized" sentences
    Inputs = tokenizer([s.split() for s in sentences],
                       padding=True,
                       truncation=True,
                       return_tensors='pt',
                       is_split_into_words=True
                      )
    word_starts_from = []
    word_ends_at = []
    for i, s in enumerate(sentences):
        subword_lens = word_boundary_sanity(s)
        start_from = sentence_starts_from(Inputs.input_ids[i])
        word_stf = np.cumsum([start_from] + subword_lens[:-1])
        word_edt = word_stf[1:]
        word_edt = np.append(word_edt, word_stf[-1] + subword_lens[-1])
        # don't include truncated words
        if word_edt[-1] > len(Inputs['input_ids'][i]):
            # find where truncation occurs
            # print('truncate!')
            cut_at = np.where(np.array(word_edt)>len(Inputs['input_ids'][i]))[0][0]
            word_stf = word_stf[:cut_at]
            word_edt = word_edt[:cut_at]

        word_starts_from.append(word_stf)
        word_ends_at.append(word_edt)
    if return_sentence:
        return Inputs.to('cuda'), word_starts_from, word_ends_at, sentences 
    else:
        return Inputs.to('cuda'), word_starts_from, word_ends_at


def one_pass_data_batches(data_path, tokenizer, batch_size,
                          bpe_tok=False, return_sentence=False):
    """
        A pass of the file. Group consecutive lines (sentences) into batch,
        and return
    """
    batch = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            batch.append(line.strip())
            if len(batch) == batch_size: # ready to yield batch
                yield create_batch(batch, tokenizer, bpe_tok, return_sentence)
                batch = []
    if len(batch) > 0:
        yield create_batch(batch, tokenizer, bpe_tok, return_sentence)
