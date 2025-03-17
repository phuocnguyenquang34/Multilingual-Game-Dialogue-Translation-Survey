from abc import ABC, abstractmethod
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

model_cache_dir = "cache/transformers"
class GameDialogueTranslator(ABC):
    def __init__(self, variation):
        """
        Initializes the GameDialogueTranslator with a specified variation.
        
        :param variation: The variation to load the model for.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.variation = variation
        self.model, self.tokenizer = self.load_model()
    
    @abstractmethod
    def load_model(self):
        """
        This method should load and return the model based on the given variation.
        Needs to be implemented in a subclass.
        
        :return: The loaded model.
        """
        pass
    
    @abstractmethod
    def translate_text(self, text, source_lang, target_lang):
        """
        Translates the given text from the source language to the target language.
        Needs to be implemented in a subclass.
        
        :param text: The text to be translated.
        :param source_lang: The language of the input text.
        :param target_lang: The language to translate the text into.
        :return: The translated text.
        """
        pass

    @abstractmethod
    def get_possible_variations():
        """
        Returns a list of possible variations for translation models.
        Subclasses should override this method if needed.
        
        :return: A list of possible model variations.
        """
        pass

class NLLB_Translator(GameDialogueTranslator):
    def __init__(self, variation='facebook/nllb-200-distilled-600M'):
        super().__init__(variation)
    
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model.to(self.device)
        return model, tokenizer
    
    def translate_text(self, text, source_lang="eng_Latn", target_lang="vie_Latn"):
        if target_lang == "Portuguese":
            target_lang = "por_Latn"
        elif target_lang == "Bengali":
            target_lang = "ben_Beng"
        elif target_lang == "Spanish":
            target_lang = "spa_Latn"
        elif target_lang == "Vietnamese":
            target_lang = "vie_Latn"
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)  # Assuming GPU usage
        inputs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(target_lang)

        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        start_time = time.time()
        output_tokens = self.model.generate(**inputs, max_length=512)
        # Record GPU memory usage after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0  # No GPU usage if CUDA is unavailable
        time_cost = time.time() - start_time
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True), time_cost, gpu_memory_used

    def get_possible_variations():
        return [
            'facebook/nllb-200-distilled-600M', 
            'facebook/nllb-200-distilled-1.3B',
            'facebook/nllb-200-3.3B'
        ]

class Qwen_Translator(GameDialogueTranslator):
    def __init__(self, variation='Qwen/Qwen2.5-0.5B-Instruct', system_prompt='You are a helpful assistant that translates game dialogues.'):
        self.system_prompt = system_prompt
        super().__init__(variation)
    
    def load_model(self):
        quantization_config = None
        if self.variation == type(self).get_possible_variations()[-1]:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(self.variation, 
                                                    #  attn_implementation='flash_attention_2',
                                                    #  torch_dtype=torch.bfloat16,
                                                     quantization_config=quantization_config, 
                                                     cache_dir=model_cache_dir)
        if quantization_config is None:
            model.to(self.device)
            
        return model, tokenizer
    
    def translate_text(self, text, source_lang="eng_Latn", target_lang="Vietnamese"):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"Translate this game dialogue to {target_lang}: \n'''\n{text}\n'''"})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        # Record GPU memory usage after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0  # No GPU usage if CUDA is unavailable
        time_cost = time.time() - start_time
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        translated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated_text, time_cost, gpu_memory_used

    def get_possible_variations():
        return [
            'Qwen/Qwen2.5-0.5B-Instruct', 
            'Qwen/Qwen2.5-1.5B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct'
        ]

class Aya_Translator(GameDialogueTranslator):
    def __init__(self, variation='CohereForAI/aya-expanse-8b', system_prompt='You are a helpful assistant that translates game dialogues.'):
        self.system_prompt = system_prompt
        super().__init__(variation)
    
    def load_model(self):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(self.variation, quantization_config=quantization_config, cache_dir=model_cache_dir)        
        return model, tokenizer
    
    def translate_text(self, text, source_lang="eng_Latn", target_lang="Vietnamese"):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"Translate this game dialogue to {target_lang}: \n'''\n{text}\n'''"})
        
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)

        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        start_time = time.time()
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.3,
            )
        # Record GPU memory usage after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0  # No GPU usage if CUDA is unavailable
        time_cost = time.time() - start_time
        
        translated_text = self.tokenizer.decode(gen_tokens[0, input_ids.shape[1]:],skip_special_tokens=True)
        return translated_text, time_cost, gpu_memory_used

    def get_possible_variations():
        return [
            'CohereForAI/aya-expanse-8b'
        ]
    
class Bloomz_mT0_Translator(GameDialogueTranslator):
    def __init__(self, variation='bigscience/mt0-xl'):
        super().__init__(variation)
    
    def load_model(self):    
        if self.variation == type(self).get_possible_variations()[0]:
            tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(self.variation, cache_dir=model_cache_dir)
        else:    
            tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model.to(self.device)
        return model, tokenizer
    
    def translate_text(self, text, source_lang="eng_Latn", target_lang="Vietnamese"):
        inputs = self.tokenizer.encode(f"Given this game dialogue: \n'''\n{text}\n'''\nTranslate to {target_lang}.", return_tensors="pt").to(self.device)
        
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        start_time = time.time()
        outputs = self.model.generate(inputs, max_length=512)
        # Record GPU memory usage after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0  # No GPU usage if CUDA is unavailable
        time_cost = time.time() - start_time

        if self.variation == type(self).get_possible_variations()[0]:
            translated_text = self.tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
        else:
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text, time_cost, gpu_memory_used

    def get_possible_variations():
        return [
            "bigscience/bloomz-3b",
            'bigscience/mt0-xl'
        ]
    
class Llama_Translator(GameDialogueTranslator):
    def __init__(self, variation='/kaggle/input/llama-3.2/transformers/1b-instruct/1', system_prompt='You are a helpful assistant that translates game dialogues.'):
        self.system_prompt = system_prompt
        super().__init__(variation)
    
    def load_model(self):
        quantization_config = None
        if self.variation in type(self).get_possible_variations()[2:]:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        tokenizer = AutoTokenizer.from_pretrained(self.variation, cache_dir=model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(self.variation, quantization_config=quantization_config, cache_dir=model_cache_dir)
        if quantization_config is None:
            model.to(self.device)
            
        return model, tokenizer
    
    def translate_text(self, text, source_lang="eng_Latn", target_lang="Vietnamese"):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"Translate this game dialogue to {target_lang}: \n'''\n{text}\n'''"})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3
        )
        # Record GPU memory usage after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0  # No GPU usage if CUDA is unavailable
        time_cost = time.time() - start_time
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        translated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated_text, time_cost, gpu_memory_used

    def get_possible_variations():
        return [
            'meta-llama/Llama-3.2-1B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct'
        ]