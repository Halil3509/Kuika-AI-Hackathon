
from graph import app
from pprint import pprint

def predict(text: str):
    inputs = {"question": text}

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")

    if 'generation' not in value.keys():
        return value['condition']
    else:
        return value["generation"]