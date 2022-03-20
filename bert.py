import torch
from keybert import KeyBERT

count = 0


def generate_sentence_embedding(sentence, tokenizer, bert_model):
    word_pieces = "[CLS]" + sentence + "[SEP]"
    sentence_tokens = tokenizer.tokenize(word_pieces)
    if len(sentence_tokens) > 512:  # truncate the sentence if it is longer than 512
        sentence_tokens = sentence_tokens[:512]
    token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)  # token_ids is a list
    segments_ids = [1] * len(sentence_tokens)
    input_tensor = torch.tensor([token_ids])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        output = bert_model(input_tensor, segments_tensors)
        hidden_layer = output.hidden_states
    token_embeddings = torch.stack(hidden_layer, dim=0)  # [# layers, # batches, # tokens, # features]
    token_embeddings = torch.squeeze(token_embeddings,
                                     dim=1)  # get rid of the “batches” dimension, [# layers, # tokens, # features]
    sentence_embedding = token_embeddings[-2, :, :]
    sentence_embedding = torch.sum(sentence_embedding, keepdims=True, dim=0)
    sentence_embedding = sentence_embedding.numpy()  # convert tensors to numpy array to fit the scikit-learn library
    return sentence_embedding


def keyword_extraction(doc):
    model = KeyBERT()
    keywords = model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), use_mmr=True, diversity=0.1)
    return keywords
