import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score
import logging
import os
from tqdm import tqdm

def get_nan_translation(df: pd.DataFrame):
    nan_cells = []
    for col in df.columns:
        for idx in df.index:
            if pd.isna(df.loc[idx, col]):
                nan_cells.append((idx, col))
    return nan_cells

def load_output(lang):
    gem_df = pd.read_csv(f"gemini_output/gemini_{lang.lower()}_game_translation_250.csv", index_col=0)
    output_df = pd.read_csv(f"output/{lang.lower()}_game_translation_250.csv", index_col=0)
    # Bloomz 3b output empty strings as translations sometimes
    output_df = output_df.fillna('')
    return gem_df, output_df

def create_bleu_chrf_dataframe(candidate_models):
    # Create multi-level column index to store detailed BLEU scores
    columns = pd.MultiIndex.from_tuples([
        ('BLEU', 'score'),
        ('BLEU', 'hyp_ref_len_ratio'),
        ('BLEU', '1-gram precision'),
        ('BLEU', '2-gram precision'),
        ('BLEU', '3-gram precision'),
        ('BLEU', '4-gram precision'),
        ('CHRF++', ''),  # No sub-columns for CHRF++
        ('Mean_BERTScore_F1', '')
    ])

    df = pd.DataFrame(columns=columns, index=candidate_models)

    return df

if __name__ == "__main__":
    lang_list = ["Vietnamese", "Portuguese", "Bengali", "Spanish"]
    save_dir = 'evaluation'
    os.makedirs(save_dir, exist_ok=True)

    for lang in tqdm(lang_list):
        logging.info(f"Processing {lang}...")
        # Load data
        gem_df, output_df = load_output(lang)
        candidate_models = output_df.index.to_list()

        # Load evaluation metric
        bleu, chrf = BLEU(), CHRF(word_order=2)

        # Create evaluation dataframe
        eval_df = create_bleu_chrf_dataframe(candidate_models)

        refs = gem_df['gemini_translation'].to_list()
        for model in candidate_models:
            trans = output_df.loc[model].to_list()[:-2]
            bleu_score = bleu.corpus_score(hypotheses=trans, references=[refs])
            chrf_score = chrf.corpus_score(hypotheses=trans, references=[refs])
            P, R, F1 = score(trans, refs, model_type="bert-base-multilingual-cased", verbose=True)
            mean_bertscore_f1 = float(F1.mean())
            results = [bleu_score.score, bleu_score.sys_len/bleu_score.ref_len, *bleu_score.precisions, chrf_score.score, mean_bertscore_f1]
            results = [round(result, 2) for result in results]
            eval_df.loc[model] = results
        eval_df.to_excel(f"evaluation/{lang.lower()}_gemini.xlsx")