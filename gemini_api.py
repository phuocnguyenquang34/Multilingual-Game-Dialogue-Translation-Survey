import base64
import os
from utils import *
from google import genai
from google.genai import types

from collections import defaultdict
import random
import time
from tqdm.notebook import tqdm

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

model = "gemini-2.0-flash"
generate_content_config = types.GenerateContentConfig(
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="text/plain",
    system_instruction=[
        types.Part.from_text(text=system_prompt),
    ],
)

def get_gemini_translation(text, target_lang):
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Translate this game dialogue to {target_lang}: \n'''\n{text}\n'''"),
            ],
        ),
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text

dataset = create_dataset(num_samples_each_depth=50)
save_dir = 'output'
os.makedirs(save_dir, exist_ok=True)

for language in ["Portuguese", "Bengali", "Spanish", "Vietnamese"]:
    print(f"_____________Language {language}")
    results = defaultdict(dict)
    for sample in tqdm(dataset):
        original_game_dialogue = sample['dialogue_tree']
        results[original_game_dialogue] = get_gemini_translation(original_game_dialogue, language)

        # Sleep
        sleep_time = random.uniform(4, 6)
        time.sleep(sleep_time)

    df = pd.DataFrame(results.items(), columns=['original', 'gemini_translation'])
    df.to_csv(f"{save_dir}/gemini_{language.lower()}_game_translation_{len(dataset)}.csv")