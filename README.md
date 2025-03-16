# LRL Game Dialogue Translator

## Project Overview

The **LRL Game Dialogue Translator** project aims to evaluate the performance of various multilingual large language models (mLLMs) in translating game dialogues. The goal is to assess whether these models can support non-English-speaking users, enabling them to enjoy game content without fluency in English.

## File Structure

### `translator.py`
This file contains the classes for different translators used in the project. Each translator class is designed to handle specific aspects of game dialogue translation using different multilingual models.

### `utils.py`
The `utils.py` file includes:
- **System prompts**: Predefined prompts to guide the models in generating accurate translations.
- **Dataset creation functions**: Functions that help create datasets for use in translation tasks and evaluations.

### `main.py`
The `main.py` file is responsible for generating translations from all translator classes. It utilizes the dataset created in `utils.py` to perform the translations, which are then used in the subsequent evaluation process.

## Purpose

The project evaluates the effectiveness of mLLMs in translating video game dialogues, aiming to make game content more accessible to players who are not fluent in English. Through this evaluation, we hope to understand how well these models can preserve the meaning, tone, and context of game dialogues in different languages.
