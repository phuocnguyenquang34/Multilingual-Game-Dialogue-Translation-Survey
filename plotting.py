import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

def read_eval_xlsx_file(file_path: str, language: str) -> pd.DataFrame:
    """
    Reads evaluation data from an Excel file.

    Input:
    - file_path: Path to the Excel file (in the evaluation folder).

    Output:
    - DataFrame: A MultiIndex DataFrame with 2 levels of columns for BLEU scores and 1 level for CHRF++ and Mean_BERTScore_F1.
    """
    df = pd.read_excel(file_path, index_col=0, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(a, b if not b.startswith('Unnamed') else '') for a, b in df.columns]
    )
    df["Language"] = language
    return df

def concat_results(evaluation_dir: str) -> pd.DataFrame:
    results_df = pd.DataFrame()
    for file in os.listdir(evaluation_dir):
        if file.endswith(".xlsx"):
            file_path = os.path.join(evaluation_dir, file)
            language = file.split("_")[0].capitalize()

            # Skip Portuguese for now since we don't have human evaluator.
            if language == "Portuguese":
                continue
            
            df = read_eval_xlsx_file(file_path, language)
        results_df = pd.concat([results_df, df], axis=0)

    results_df = results_df.reset_index()
    results_df.columns = ['Model', 'BLEU', 'hyp_ref_len_ratio', '1-gram', '2-gram', '3-gram', '4-gram',
                        'CHRF++', 'Mean_BERTScore_F1', 'Language']
    return results_df
    
def extract_param_size(model_name):
    """
    Extracts parameter size from model name.
    Returns float number in billions (e.g., 0.6 for 600M, 3.3 for 3.3B).
    """
    match = re.search(r'(\d+(?:\.\d+)?)([MB])', model_name)
    if match:
        num, unit = match.groups()
        num = float(num)
        return num / 1000 if unit == 'M' else num  # Convert M → B
    return None

def create_plot_topk_results(df_long: pd.DataFrame, top_K: int = 5):
    metrics = df_long["Metric"].unique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_metric = df_long.loc[df_long['Metric'] == metric]
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

        # Step 1: Top-K filtering
        df_metric = df_metric.sort_values(['Language', 'Score'], ascending=[True, False])
        df_metric = df_metric.groupby('Language', group_keys=False).head(top_K)

        # Step 2: Determine order by average score
        order = df_metric.groupby("Language")["Score"].mean().sort_values(ascending=False).index

        # Step 3: Apply categorical ordering
        print(order)
        df_metric['Language'] = pd.Categorical(df_metric['Language'], categories=order, ordered=True)
        df_metric = df_metric.sort_values('Language')

        # Step 4: Compute and apply the same ordering to mean line
        mean_scores = df_metric.groupby('Language')['Score'].mean().reset_index()
        mean_scores['Model'] = 'Language Mean'
        mean_scores['Language'] = pd.Categorical(mean_scores['Language'], categories=order, ordered=True)
        mean_scores = mean_scores.sort_values('Language')

        # Plot the mean as a dotted black line
        sns.lineplot(
            data=mean_scores, 
            x='Language', 
            y='Score', 
            label='mean', 
            ax=ax, 
            linestyle='--', 
            color='black', 
            linewidth=1.2)
        
        sns.lineplot(
            data=df_metric, 
            x='Language', 
            y='Score', 
            hue='Model', 
            marker='o', 
            ax=ax, 
            linewidth=1.5, 
            )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(metric, fontsize=13, weight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.legend_.remove() 

        # Highlight best per language
        for lang in df_metric['Language'].unique():
            best_row = df_metric[df_metric['Language'] == lang].sort_values(by='Score', ascending=False).iloc[0]
            ax.scatter(
                x=lang,
                y=best_row['Score'],
                s=100,
                edgecolor='black',
                facecolor='none',
                linewidth=1.5,
                marker='o',
                zorder=5
            )

    # Shared x and y labels
    fig.supxlabel("Language", fontsize=12, y=0.05)
    fig.supylabel("Score", fontsize=12, x=0.01)
    # plt.suptitle(f"Top {top_K} Models Performance Across Languages", fontsize=16, weight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('images/top_5_models.png', bbox_inches='tight')

def extract_family(model_name):
    if "nllb" in model_name:
        return "NLLB"
    elif "llama" in model_name.lower():
        return "LLama"
    elif "qwen" in model_name.lower():
        return "Qwen"
    elif "bloomz" in model_name.lower():
        return "Bloom"
    elif "aya" in model_name.lower():
        return "Aya"
    elif "mt0" in model_name.lower():
        return "mt0"
    else:
        return "Other"    

def process_family_data(df_long: pd.DataFrame):
    df_long["Family"] = df_long["Model"].apply(extract_family)
    filtered_family_df = df_long[df_long['Family'].isin(multi_varients_family)]
    all_family_df = pd.DataFrame()

    for family in multi_varients_family:
        family_df = filtered_family_df[filtered_family_df['Family']==family]
        family_df['Score_normalized'] = (
            filtered_family_df.groupby('Metric')['Score']
                .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
        family_df = (
            family_df.groupby('Model')['Score_normalized']
                .mean()
                .reset_index()
                .rename(columns={'Score_normalized': 'Normalized_Mean_Score'})
        )
        family_df['Family'] = family
        all_family_df = pd.concat([all_family_df, family_df], axis=0)

        all_family_df['Param_Size'] = all_family_df['Model'].apply(extract_param_size)
    df_grouped = (
        all_family_df.groupby(['Family', 'Param_Size'], as_index=False)['Normalized_Mean_Score']
        .mean()
    )
    return df_grouped

def create_plot_family_tradeoff(df_long):
    df_grouped = process_family_data(df_long)
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Line plot by Family vs Param_Size 
    ax = sns.lineplot(
        data=df_grouped,
        x="Param_Size",
        y="Normalized_Mean_Score",
        hue="Family",
        marker="o",
        linewidth=2
    )

    for i, row in df_grouped.iterrows():
        ax.text(
            row['Param_Size'],
            row['Normalized_Mean_Score'] + 0.025,  # move label 
            f"{row['Param_Size']}B",
            fontsize=8,
            ha='center',
        )

    # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.title("Normalized Mean Score vs Parameter Size by Family", fontsize=14, weight='bold')
    plt.grid(False)
    plt.xlabel("Parameter Size (Billions)", fontsize=12)
    plt.ylabel("Normalized Mean Score", fontsize=12)
    plt.tight_layout()
    plt.savefig('images/performance_vs_size.png', bbox_inches='tight')

if __name__ == "__main__":
    evaluation_dir = "evaluation"
    top_K = 5
    multi_varients_family = ['NLLB', 'LLama', 'Qwen']

    results_df = concat_results(evaluation_dir)

    df_long = results_df.melt(id_vars=['Model', 'Language'], 
                            value_vars=['BLEU', 'CHRF++', 'Mean_BERTScore_F1'],
                            var_name='Metric', value_name='Score')
    create_plot_topk_results(df_long, top_K)
    create_plot_family_tradeoff(df_long)

