import streamlit as st
import pandas as pd
import os
import random

import base64

def set_top_background(image_file, height="200px"):
    """
    Sets the background of the top portion of the Streamlit app to an image.

    Args:
        image_file (str): Path to the image file.
        height (str): Height of the background area (e.g., "200px", "30%").
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .top-background {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            height: {height};
            width: 100%;
        }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)
    st.markdown('<div class="top-background"></div>', unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    set_top_background('images/top_background.jpg') 
    st.title("Game Dialogues Translation Evaluation")

    data_dir = "gemini_output" 
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
        language = file_name.split("_")[1].capitalize()
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            dataframes[language] = df
            language_list.append(language)
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")

    if dataframes:
        selected_csv = st.selectbox("Select Translation Language:", language_list)

        if selected_csv:
            df = dataframes[selected_csv]

            if 'original' not in df.columns or 'gemini_translation' not in df.columns:
                st.error("CSV file must contain 'question' and 'answer' columns.")
                return

            if 'question_indices' not in st.session_state:
                st.session_state.question_indices = random.sample(range(len(df)), min(5, len(df)))
                st.session_state.current_index = 0
                st.session_state.ratings = {} # Initialize ratings storage

            if st.session_state.current_index < len(st.session_state.question_indices):
                index = st.session_state.question_indices[st.session_state.current_index]

                original = df['original'].iloc[index]
                translation = df['gemini_translation'].iloc[index]

                st.subheader("Game Context:")
                st.markdown("""
                Four thousand years before the rise of the Galactic Empire, the Republic verges on collapse. DARTH MALAK, last surviving apprentice of the Dark Lord Revan, has unleashed an invincible Sith armada upon an unsuspecting galaxy.

                Crushing all resistance, Malak's war of conquest has left the Jedi Order scattered and vulnerable as countless Knights fall in battle, and many more swear allegiance to the new Sith Master.

                In the skies above the Outer Rim world of Taris, a Jedi battle fleet engages the forces of Darth Malak in a desperate effort to halt the Sith's galactic domination....
                """)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original:")
                    st.write(original.replace("\n", "  \n"))

                    rating = st.radio("Rate the Answer:", [1, 2, 3, 4, 5])

                with col2:
                    st.subheader("Translation:")
                    st.write(translation.replace("\n", "  \n"))

                if st.button("Submit Rating"):
                    st.session_state.ratings[index] = rating
                    st.write(f"You rated the answer {rating} ⭐.")

                if st.button("Next Question"):
                    st.session_state.current_index += 1
                    st.rerun()  # Rerun the script to load the next question
                
                st.subheader("Evaluation Rubric:")
                st.markdown("""
                Rate the translation on a scale from 1 to 5 based on two main criteria: correctness and attractiveness.
                * **1 Star:** The translation is completely inaccurate, with the **meaning diverging significantly** from the original English. The language is also **unattractive and lacks engagement**.
                * **2 Stars:** The translation **maintains a general connection to the English meaning**, but the **word choices** are **unattractive and lack flair**.
                * **3 Stars:** The translation **accurately conveys the English meaning**, but the **language** remains somewhat **basic** and **lacks emotional depth or style**.
                * **4 Stars:** The translation **stays true to the original meaning** and the **language** is **more expressive**, showing **emotional resonance or game-specific stylistic elements**, making it more engaging.
                * **5 Stars:** The translation **perfectly captures the meaning and tone** of the original English dialogue, with **highly attractive and immersive language**. It uses appropriate pronouns, emotionally rich vocabulary, and gaming-specific terminology to enhance the overall feel of the dialogue.
                """)

            else:
                st.write("All translation have been evaluated.")
                if st.session_state.ratings:
                    ratings_df = pd.DataFrame(list(st.session_state.ratings.items()), columns=['index', 'rating'])
                    ratings_df['selected_csv'] = selected_csv
                    st.write("Ratings:")
                    st.dataframe(ratings_df)

                    if st.download_button("Download Ratings as CSV", ratings_df.to_csv(index=False).encode('utf-8'), "ratings.csv", "text/csv"):
                        st.success("Ratings downloaded!")
    else:
        st.write("No valid CSV files found in the directory.")

if __name__ == "__main__":
    main()