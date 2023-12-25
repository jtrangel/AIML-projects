import json
import torch
from transformers import AutoModel, AutoTokenizer

# Soft type assertions
from typing import Dict, Tuple, List
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch import Tensor

# get the available questions and answers for a given topic
def get_qa(topic: str, data: Dict) -> Tuple[List[str], List[str]]:
    q = []
    a = []
    for d in data['data']:
        if d['title'] == topic:
            for paragraph in d['paragraphs']:
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        q.append(qa['question'])
                        a.append(qa['answers'][0]['text'])
            return q, a

# Get and define model from same directory
def get_model(model_name:str ) -> Tuple[BertModel, BertTokenizerFast]:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Mean Pooling - Take attention mask into account for correct averaging
# source: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
def mean_pooling(model_output: BaseModelOutputWithPoolingAndCrossAttentions,
                 attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0]

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    pool_emb = (
            torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )

    return pool_emb

def get_embeddings(questions: List,
                   tokenizer : BertTokenizerFast,
                   model: BertModel) -> Tensor:
    # Tokenize sentences
    encoded_input = tokenizer(questions, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Average pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return embeddings

if __name__ == "__main__":

  # Read the context data
  with open("train-v2.0.json", 'r') as f:
      data = json.load(f)

  # Extract the QnA context/facts
  questions, answers = get_qa(topic='Premier_League', data=data)
  print("Number of available questions: {}".format(len(questions)))

  # Define model
  model, tokenizer = get_model(model_name="paraphrase-MiniLM-L6-v2")

  print(type(model))
  print(type(tokenizer))

  # Get vector embeddings
  embeddings = get_embeddings(questions[:3], tokenizer, model)
  print("Embeddings shape: {}".format(embeddings.shape))

  # Embeddings test
  new_question = 'Which days have the most events played at?'
  new_embedding = get_embeddings([new_question], tokenizer, model)

  # squared Euclidean distance between sample questions and new_question
  print(((embeddings - new_embedding) ** 2).sum(axis=1))

