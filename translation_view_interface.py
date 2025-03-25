import streamlit as st
import pandas as pd
import os

def main():
    st.set_page_config(layout="wide")
    st.title("Output Translation Viewer")

    data_dir = "output"  # Directory containing CSV files
    if not os.path.exists(data_dir):
        st.error(f"Directory '{data_dir}' not found. Please create it and place CSV files inside.")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        st.warning(f"No CSV files found in '{data_dir}'.")
        return

    dataframes = {}
    language_list = []
    for file_name in csv_files:
        language = file_name.split("_")[0].capitalize()
        file_path = os.path.join(data_dir, file_name)
        language_list.append(language)
        try:
            df = pd.read_csv(file_path, index_col=0)
            dataframes[language] = df
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")

    if dataframes:
        gpu_to_load_model_df = pd.read_csv("gpu_to_load_model.csv", index_col=0)
        selected_csv = st.selectbox("Select Output File:", language_list)

        if selected_csv:
            df = dataframes[selected_csv]
            indices = list(df.index)
            selected_index = st.selectbox("Select Model:", indices)
            row = df.loc[selected_index]
            st.write(row[:-2])
            st.write(f"GPU cost to load model: {round(gpu_to_load_model_df.loc[selected_index].values[0], 4)}MB")
            st.write(f"Mean time cost: {round(row.mean_time_cost, 4)}s")
            st.write(f"Mean GPU cost each run: {round(row.mean_gpu_cost/1024.0**2, 4)}MB")
    else:
        st.write("No valid CSV files found in the directory.")

if __name__ == "__main__":
    main()