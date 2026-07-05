import json
import ollama
import outlines.models as models
from outlines.generator import Generator
from pydantic import BaseModel


class Profile(BaseModel):
    name: str
    age: int


def main():
    client = ollama.Client()
    model = models.Ollama(client, "llama3.2:3b")
    generator = Generator(model, output_type=Profile)

    result = generator("Generate a random person profile with name and age.")

    # Ollama (BlackBoxModel) returns a JSON string — parse it
    if isinstance(result, str):
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    else:
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()