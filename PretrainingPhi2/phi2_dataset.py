import json
import random
import uuid
from ollama import Client
from datasets import load_dataset, IterableDataset, Dataset
from datasets import interleave_datasets

# Connect to Ollama Server
client = Client(base_url="http://127.0.0.1:8008")

# Read the Vocab of Phi2 model
with open("vocab.json", "rb") as f:
    vocab_data = json.load(f)
ALL_TOKENS = [token for token in vocab_data.keys() if len(token)>3]
ALL_TOKENS = [token[1:] for token in ALL_TOKENS if token.startswith("Ġ")]

# Select a random token from Vocab
def get_random_token():
    return random.choice(ALL_TOKENS)

# Fetch the data from Phi2 running in Ollama server
def data_from_phi():
    word = get_random_token()
    message = {'role': 'user', 'content': f'tell me about the word {word} in 50 words'}
    response = client.chat(model='phi', messages=[message])
    content = response["message"]["content"]
    return content

# Generator util for Ollama API wrapper
class _DatasetGeneratorPickle:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickle_raise, (self.generator_id,))


def _DatasetGeneratorPickle_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickle!")

# Data Generator from Phi2
def phi_data_generator(n, *args, **kwargs):
    for i in range(n):
        content = data_from_phi()
        yield {"text": content}

# Create Iterable Dataset from Ollama Phi2 API
phi2_gen_dataset = IterableDataset.from_generator(_DatasetGeneratorPickle(phi_data_generator),
                                                  gen_kwargs={"n": 1000000})

# Create an Iterable dataset from TinyStories book corpus dataset
tiny_stories_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Combine both the datasets which sends one of the sample data with 50% probability
dataset = interleave_datasets([phi2_gen_dataset, tiny_stories_dataset])