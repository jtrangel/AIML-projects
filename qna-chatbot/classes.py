import json
import torch
from transformers import AutoModel, AutoTokenizer

# Soft type assertions
from typing import Dict, Tuple, List
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch import Tensor

class QAEmbedder:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        """
        QnA embedding model.

        For a given a set of questions, returns the corresponding embedding vectors.

        :param model_name: Directory name with model and tokenizer files
        :type model_name: str
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.set_model(model_name)

    def get_model(self, model_name: str) -> Tuple[BertModel, BertTokenizerFast]:
        """
        Get and define model for specified model name/directory

        :param model_name: model name/directory
        :type model_name: str

        :return: Model and Tokenizer objects from the directory
        :rtype: Tuple[BertModel, BertTokenizerFast]
        """
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def set_model(self, model_name: str) -> None:
        """
        Instantiates a general tokenizer and model for the class using the 'self.get_model'
        method.

        :param model_name: model name/directory
        :type model_name: str
        """
        self.model, self.tokenizer = self.get_model(self.model_name)

    # get the available questions and answers for a given topic
    def get_qa(self,
               topic: str,
               data: Dict) -> Tuple[List[str], List[str]]:
        """
        Gets the available questions and answers for a given topic (from the json context data)

        :param topic: specific topic of focus
        :type topic: str
        :param data: the json context data
        :type data: Dict

        :return: list of questions and corresponding answers
        :rtype: Tuple[List[str], List[str]]
        """
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

    def mean_pooling(self,
                     model_output: BaseModelOutputWithPoolingAndCrossAttentions,
                     attention_mask: Tensor) -> Tensor:
        """
        Mean Pooling - Take attention mask into account for correct averaging
        source: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2

        :param model_output:
        :type model_output:
        :param attention_mask:
        :type attention_mask:

        :return:
        :rtype: Tensor
        """
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

    def get_embeddings(self,
                       questions: List,
                       batch: int = 32) -> Tensor:
        """

        :param questions:
        :type questions:
        :param batch:
        :type batch: int

        :return:
        :rtype:
        """
        question_embeddings = []
        for i in range(0, len(questions), batch):
            # Tokenize sentences
            encoded_input = self.tokenizer(questions[i:i + batch], padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform mean pooling
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            question_embeddings.append(batch_embeddings)

        question_embeddings = torch.cat(question_embeddings, dim=0)
        return question_embeddings


class QASearcher:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        """
        Defines a QA Search model. This is, given a new question it searches
        the most similar questions in a set 'context' and returns both the best
        question and associated answer.

        Args:
          model_name (`str`): Directory containing the necessary tokenizer
            and model files.
        """
        self.answers = None
        self.questions = None
        self.question_embeddings = None
        self.embedder = QAEmbedder(model_name=model_name)

    def set_context_qa(self, questions, answers):
        """
        Sets the QA context to be used during search.

        Args:
          questions (`list` of `str`):  List of strings defining the questions to be embedded
          answers (`list` of `str`): Best answer for each question in 'questions'
        """
        self.answers = answers
        self.questions = questions
        self.question_embeddings = self.get_q_embeddings(questions)

    def get_q_embeddings(self, questions):
        """
        Gets the embeddings for the questions in 'context'.

        Args:
          questions (`list` of `str`):  List of strings defining the questions to be embedded

        Returns:
          The embedding vectors.
        """
        question_embeddings = self.embedder.get_embeddings(questions)
        question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
        return question_embeddings.transpose(0, 1)

    def cosine_similarity(self, questions, batch=32):
        """
        Gets the cosine similarity between the new questions and the 'context' questions.

        Args:
          questions (`list` of `str`):  List of strings defining the questions to be embedded
          batch (`int`): Performs the embedding job 'batch' questions at a time

        Returns:
          The cosine similarity
        """
        question_embeddings = self.embedder.get_embeddings(questions, batch=batch)
        question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)

        cosine_sim = torch.mm(question_embeddings, self.question_embeddings)

        return cosine_sim

    def get_answers(self, questions, batch=32):
        """
        Gets the best answers in the stored 'context' for the given new 'questions'.

        Args:
          questions (`list` of `str`):  List of strings defining the questions to be embedded
          batch (`int`): Performs the embedding job 'batch' questions at a time

        Returns:
          A `list` of `dict`'s containing the original question ('orig_q'), the most similar
          question in the context ('best_q') and the associated answer ('best_a').
        """
        similarity = self.cosine_similarity(questions, batch=batch)

        response = []
        for i in range(similarity.shape[0]):
            best_ix = similarity[i].argmax()
            best_q = self.questions[best_ix]
            best_a = self.answers[best_ix]

            response.append(
                {
                    'orig_q': questions[i],
                    'best_q': best_q,
                    'best_a': best_a,
                }
            )

        return response