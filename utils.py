import torch
from datasets import Dataset

import gc
import pandas as pd
import json

system_prompt = """You are a professional game translator, whose goal is \
to ensure players speaking languages other than English can fully experience the game as the authors' \
intention. The game you are currently focusing on right now is Star Wars: Knights of the Old Republic. 

The game's content is as follow:
'''
Four thousand years before the rise of the Galactic Empire, the Republic verges on collapse. DARTH \
MALAK, last surviving apprentice of the Dark Lord Revan, has unleashed an invincible Sith armada upon \
an unsuspecting galaxy.

Crushing all resistance, Malak's war of conquest has left the Jedi Order scattered and vulnerable as \
countless Knights fall in battle, and many more swear allegiance to the new Sith Master.

In the skies above the Outer Rim world of Taris, a Jedi battle fleet engages the forces of Darth Malak \
in a desperate effort to halt the Sith's galactic domination....
'''

Your task is given the game dialogues, you should refer to the game's content to select words' \
translation that best convey the game style dialogues. Also, several emotions that can be found within \
the dialogues such as anger, happiness, sad, hatred should also be demonstrated in the translation as\
well, so that the players can understand the rhythm and pace of the game. The stability in pronouns \
is also one of the critic aspect.

Only answer with your translation, no explaination is needed."""

def clean_gpu():
    # Invoke garbage collector
    gc.collect()
    # Clear GPU cache
    torch.cuda.empty_cache()

def generate_dialogue_trees(data, max_dialogue_depth=5):
    dialogues_data = []
    traversed_rows = set()

    for index, row in data.iterrows():
        if index in traversed_rows:
            continue

        dialogue_tree = [f"{row['speaker']}: {row['text']}"]
        traversed_rows.add(index)
        depth = 1
        current_row = row

        while depth < max_dialogue_depth:
            next_choice_ids = json.loads(current_row['next'])
            dialogue_choice_id = next((choice_id for choice_id in next_choice_ids if choice_id not in traversed_rows), None)
            
            if dialogue_choice_id is None:
                break

            current_row = data.loc[dialogue_choice_id]
            dialogue_tree.append(f"{current_row['speaker']}: {current_row['text']}")
            traversed_rows.add(dialogue_choice_id)
            depth += 1

        dialogues_data.append({"dialogue_tree": "\n".join(dialogue_tree), "depth": depth})
    
    return pd.DataFrame(dialogues_data)

def create_dataset(max_dialogue_depth=5, num_samples_each_depth=200, random_state=42):
    data = pd.read_csv("/home/leelab-alignfreeze2/LRL-Game-Dialogue-Translator/dataset_20200716.csv", index_col="id")
    # Remove unecessary cols
    data = data.drop(["listener", "animation", "comment", "previous", "source_dlg", "audiofile"], axis=1)
    print(f"Total duplicated records in csv file: {sum(data.duplicated())}")
    dialogue_dataset = generate_dialogue_trees(data, max_dialogue_depth=max_dialogue_depth)
    print(f"Total duplicated records in dataset: {sum(dialogue_dataset.duplicated())}")
    dialogue_dataset = dialogue_dataset.drop_duplicates()
    print(f"Total unique records in dataset: {len(dialogue_dataset)}")
    print(dialogue_dataset.depth.value_counts())
    selected_examples = (
        dialogue_dataset.groupby("depth", group_keys=False)
        .apply(lambda x: x.sample(n=num_samples_each_depth, random_state=random_state) if not x.empty else None)
    )
    selected_examples.reset_index(drop=True)
    return Dataset.from_pandas(selected_examples, preserve_index=False)