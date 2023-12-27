import json
from classes import QAEmbedder, QASearcher

emb = QAEmbedder()
search = QASearcher()

# Read the context data
with open("../train-v2.0.json", 'r') as f:
  data = json.load(f)

# Extract the QnA context/facts
questions, answers = emb.get_qa(topic='Premier_League', data=data)
print("Number of available questions: {}".format(len(questions)))

# Define model
model, tokenizer = emb.get_model(model_name="paraphrase-MiniLM-L6-v2")

print(type(model))
print(type(tokenizer))

# Get vector embeddings
embeddings = emb.get_embeddings(questions[:3])
print("Embeddings shape: {}".format(embeddings.shape))

# Embeddings test
new_question = 'Which days have the most events played at?'
new_embedding = emb.get_embeddings([new_question])

# squared Euclidean distance between sample questions and new_question
print(((embeddings - new_embedding) ** 2).sum(axis=1))

new_questions = [
    'How many teams compete in the Premier League ?',
    'When does the Premier League starts and finishes ?',
    'Highest number of goals in the Premier League ?',
]

search.set_context_qa(
  questions,
  answers
)
response = search.get_answers(questions=new_questions)

print(json.dumps(response, indent=4, sort_keys=True))

