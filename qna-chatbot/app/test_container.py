import requests
import json
from classes import QAEmbedder

def test(url: str) -> None:
    emb = QAEmbedder()

    # Load data then set the context using POST
    with open("../train-v2.0.json", 'r') as f:
        data = json.load(f)

    questions, answers = emb.get_qa(topic='Premier_League', data=data)

    json_data = {
      'questions': questions,
      'answers': answers,
    }

    response = requests.post(
      f'{url}/set_context',
      json=json_data
    )

    print(response.json())

    new_questions = [
        'How many teams compete in the Premier League ?',
        'When does the Premier League starts and finishes ?',
        'Who has the highest number of goals in the Premier League ?',
    ]

    json_data = {
      'questions': new_questions,
    }

    response = requests.post(
      f'{url}/get_answer',
      json=json_data
    )

    for d in response.json():
        print('\n'.join(["{} : {}".format(k, v) for k, v in d.items()])+'\n')

if __name__ == '__main__':
    # Test local server
    # test("http://127.0.0.1:8000")

    # Test GCP VM
    test("http://34.16.173.104:8000")
