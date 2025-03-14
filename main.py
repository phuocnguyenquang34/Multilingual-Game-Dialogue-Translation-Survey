from collections import defaultdict
from tqdm import tqdm
from utils import clean_gpu, system_prompt
from translator import *

for language in ["Portuguese", "Bengali", "Spanish"]:
    print(f"_____________Language {language}")
    results = defaultdict(dict)
    
    # Iterate through each subclass in the translator class list
    for subclass in translator_class_list:
        variation_list = subclass.get_possible_variations()
        
        for variation in tqdm(variation_list):
            total_time = 0
            variation_result = defaultdict(dict)
            
            print(f"_____Working on {variation}")
            
            # Attempt to initialize the translator with system_prompt if required
            try:
                translator = subclass(variation=variation, system_prompt=system_prompt)
            except TypeError:
                # For translators that don't require the system_prompt (e.g., NLLB, Bloomz, mT0)
                translator = subclass(variation=variation)
            
            # Translate each selected example
            for example in selected_examples:
                original_game_dialogue = example['dialogue_tree']
                
                # Start timing the translation
                translation, time_cost = translator.translate_text(original_game_dialogue, target_lang=language)
                clean_gpu()
                total_time += time_cost
                
                # Save the result
                variation_result[original_game_dialogue] = translation
            
            # Calculate the mean time cost
            variation_result['mean_time_cost'] = total_time / len(selected_examples)
            
            # Append the result for this variation
            results[variation] = variation_result
            
            del translator.model
            clean_gpu()
            print(f"Example: \n{original_game_dialogue} \n\n{translation}")
    df = pd.DataFrame(results).T
    df.to_csv(f"{language.lower()}_game_translation_test.csv")