import torch
import torch.nn as nn
from transformers import *


ckpts = [
          # Autoencoding models
          'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased',
          'albert-base', 'albert-large',
          'roberta-base', 'roberta-large',
          'distilbert-base-uncased', 'distilbert-base-cased', 'distilbert-base-multilingual-cased',
          'xlm-mlm-17-1280', 'xlm-mlm-100-1280',
          'xlm-mlm-ende-1024', 'xlm-clm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024',
          'xlm-roberta-base', 'xlm-roberta-large',
          'longformer-base',

          # causal models
          'gpt', 'gpt2', 'gpt2-medium', 'gpt2-large',
          'xlnet-base-cased', 'xlnet-large-cased',

          # seq-to-seq models
          'bart-base', 'bart-large', 'bart-large-cnn',
          't5-small', 't5-base', 't5-large', 't5-3b',

          # already trained/finetuned for certain downstream tasks
          # sentence classification
          'roberta-large-mnli'
          ]

def load_ckpt(name):
    """
    Return handle over some commonly used huggingface ckpts and tokenizers
    Refer to https://huggingface.co/transformers/model_summary.html 
    a complete list of models, here we only look at a subset of them

    get features by
    model(
        **tokenizer(
                    list_of_sentences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
    ).last_hidden_state
    """
    bpe_tok = False  # tricky tokenizer

    # Autoencoding models
    # BERTs: [CLS], ..., [SEP]
    if name == 'bert-base-uncased':
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    elif name == 'bert-large-uncased':
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    elif name == 'bert-base-cased':
        model = BertModel.from_pretrained("bert-base-cased")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
    elif name == 'bert-large-cased':
        model = BertModel.from_pretrained("bert-large-cased")
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

    # alberts: [CLS], ..., [SEP]
    elif name == 'albert-base':
        model = AlbertModel.from_pretrained('albert-base-v2')
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    
    elif name == 'albert-large':
        model = AlbertModel.from_pretrained('albert-large-v2')
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')

    elif name == 'albert-xxlarge':
        model = AlbertModel.from_pretrained('albert-xxlarge-v2')
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')

    # Robertas: <s>, ..., </s>
    elif name == 'roberta-base':
        model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        bpe_tok = True

    elif name == 'roberta-large':
        model = RobertaModel.from_pretrained('roberta-large')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        bpe_tok = True

    # distilBerts: [CLS], ..., [SEP]
    elif name == 'distilbert-base-uncased':
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    elif name == 'distilbert-base-cased':
        model = DistilBertModel.from_pretrained('distilbert-base-cased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    elif name == 'distilbert-base-multilingual-cased':
        model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    # XLMs: <s>, ..., </s>
    elif name == 'xlm-mlm-17-1280':
        model = XLMModel.from_pretrained('xlm-mlm-17-1280')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-17-1280')

    elif name == 'xlm-mlm-100-1280':
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')

    elif name == 'xlm-mlm-en-2048':
        model = XLMModel.from_pretrained('xlm-mlm-en-2048')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

    elif name == 'xlm-mlm-ende-1024':
        model = XLMModel.from_pretrained('xlm-mlm-ende-1024')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-ende-1024')

    elif name == 'xlm-clm-ende-1024':
        model = XLMModel.from_pretrained('xlm-clm-ende-1024')
        tokenizer = XLMTokenizer.from_pretrained('xlm-clm-ende-1024')

    elif name == 'xlm-mlm-enfr-1024':
        model = XLMModel.from_pretrained('xlm-mlm-enfr-1024')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024')

    elif name == 'xlm-clm-enfr-1024':
        model = XLMModel.from_pretrained('xlm-clm-enfr-1024')
        tokenizer = XLMTokenizer.from_pretrained('xlm-clm-enfr-1024')

    elif name == 'xlm-mlm-enro-1024':
        model = XLMModel.from_pretrained('xlm-mlm-enro-1024')
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enro-1024')
    
    # XLM-Robertas: <s>, ..., </s>
    elif name == 'xlm-roberta-base':
        model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    elif name == 'xlm-roberta-large':
        model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

    # Funnel-transformer: <cls>, ..., <sep>
    elif name == 'funnel-small':
        model = FunnelModel.from_pretrained("funnel-transformer/small")
        tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/small")

    elif name == 'funnel-medium':
        model = FunnelModel.from_pretrained("funnel-transformer/medium")
        tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/medium")

    elif name == 'funnel-intermediate':
        model = FunnelModel.from_pretrained("funnel-transformer/intermediate")
        tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/intermediate")
    
    elif name == 'funnel-large':
        model = FunnelModel.from_pretrained("funnel-transformer/large")
        tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/large")

    elif name == 'funnel-xlarge':
        model = FunnelModel.from_pretrained("funnel-transformer/xlarge")
        tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge")

    # longformers: <s>, ..., </s>
    elif name == 'longformer-base':
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        bpe_tok = True

    elif name == 'longformer-large':
        model = LongformerModel.from_pretrained('allenai/longformer-large-4096')
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
        bpe_tok = True

    # Autoregressive models
    # GPTs: no special token inserted or appended
    elif name == 'gpt':
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    elif name == 'gpt2':
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        bpe_tok = True

    elif name == 'gpt2-medium':
        model = GPT2Model.from_pretrained('gpt2-medium')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        bpe_tok = True

    elif name == 'gpt2-large':
        model = GPT2Model.from_pretrained('gpt2-large')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        bpe_tok = True

    elif name == 'gpt2-xl':
        model = GPT2Model.from_pretrained('gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        bpe_tok = True

    # transformer-xl: no special token inserted or appended
    elif name == 'transfo-xl-wt103':
        model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # XLnets: ..., <sep>, <cls>
    elif name == 'xlnet-base-cased': 
        model = XLNetModel.from_pretrained('xlnet-base-cased')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    elif name == 'xlnet-large-cased':
        model = XLNetModel.from_pretrained('xlnet-large-cased')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')


    # Sequence-to-sequence models
    # BARTs:
    elif name == 'bart-base':
        model = BartModel.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        bpe_tok = True

    elif name == 'bart-large':
        model = BartModel.from_pretrained('facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        bpe_tok = True

    elif name == 'bart-large-cnn':
        model = BartModel.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        bpe_tok = True

    # T5s: ..., </s>
    elif name == 't5-small':
        model = T5EncoderModel.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')

    elif name == 't5-base':
        model = T5EncoderModel.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

    elif name == 't5-large':
        model = T5EncoderModel.from_pretrained('t5-large')
        tokenizer = T5Tokenizer.from_pretrained('t5-large')

    elif name == 't5-3b':
        model = T5EncoderModel.from_pretrained('t5-3b')
        tokenizer = T5Tokenizer.from_pretrained('t5-3b')

    elif name == 't5-11b':
        model = T5EncoderModel.from_pretrained('t5-11b')
        tokenizer = T5Tokenizer.from_pretrained('t5-11b')

    # already train/finetuned for certain downstream tasks
    elif name == 'roberta-large-mnli':
        model = RobertaModel.from_pretrained('roberta-large-mnli')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        bpe_tok = True

    return model.to('cuda'), tokenizer, bpe_tok


if __name__ == '__main__':
    for ckpt in ckpts:
        print(ckpt)
