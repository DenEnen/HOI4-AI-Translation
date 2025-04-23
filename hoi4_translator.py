# ... (Keep imports and previous functions like is_inside_quotes, load_glossary, load_yaml, write_yaml, post_process_output_file, preprocess_text, restore_text, translate_batch) ...
import argparse
import csv
import re
import sys
import yaml # Keep for error types if needed
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from concurrent.futures import ThreadPoolExecutor # Keep for potential future use
import math

# Attempt to import necessary libraries and provide helpful errors if missing
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Error: Missing required libraries. Please install transformers and torch:")
    print("pip install transformers torch PyYAML")
    sys.exit(1)

try:
    # Check if PyYAML is importable, though we avoid its loader for main parsing
    import yaml
except ImportError:
    print("Error: Missing required library PyYAML. Please install it:")
    print("pip install PyYAML")
    sys.exit(1)


# --- Helper Function for Quote Checking (for Glossary) ---
def is_inside_quotes(text, index):
    # ... (no changes needed)
    quote_count = 0
    i = 0
    while i < index:
        char = text[i]
        if char == '"':
            bs_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                bs_count += 1
                j -= 1
            if bs_count % 2 == 0:
                 quote_count += 1
        i += 1
    return quote_count % 2 != 0

# --- Core Translation Logic ---

def load_glossary(filepath):
    # ... (no changes needed)
    glossary = {}
    if not filepath: return glossary
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # ... (rest of the function)
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) == 2:
                    source_term, target_term = row
                    if source_term.strip():
                        glossary[source_term.strip()] = target_term.strip()
                else:
                    print(f"Warning: Skipping invalid row {i+1} in glossary (expected 2 columns): {row}", file=sys.stderr)
        print(f"Loaded {len(glossary)} terms from glossary: {filepath}")
        return dict(sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True))
    except FileNotFoundError:
        print(f"Error: Glossary file not found: {filepath}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error loading glossary: {e}", file=sys.stderr)
        return {}


def load_yaml(filepath, is_vanilla_file=False):
    """
    Loads YAML data from a file.
    Handles both 'KEY:0 "Value"' and 'KEY: "Value"' formats.
    Returns lang_key (or None), data_dictionary (or None on error).
    Strips leading/trailing whitespace from values.
    """
    lang_key = None
    inner_dict = {}
    line_num = 0
    try:
        # Use utf-8-sig to handle potential BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            line_num += 1
            if not first_line:
                print(f"Warning: File is empty: {filepath}", file=sys.stderr)
                return None, {}

            if first_line.startswith('\ufeff'):
                bom_stripped_line = first_line.lstrip('\ufeff').strip()
            else:
                bom_stripped_line = first_line.strip()

            match = re.match(r'^\s*(l_[a-zA-Z_]+)\s*:?\s*$', bom_stripped_line)
            if match:
                lang_key = match.group(1)
            else:
                f.seek(0) # Rewind to parse first line as data
                line_num = 0 # Reset line number for parsing
                if not is_vanilla_file:
                     print(f"Warning: First line in {filepath} ('{bom_stripped_line}') does not match 'l_language:'. Parsing values directly.", file=sys.stderr)
                lang_key = "l_unknown" # Assign placeholder

            # ***** MODIFIED REGEX *****
            # Makes the index group `(?:\s*\d+)?` optional.
            # Key is group 1, Value part is group 2.
            line_pattern = re.compile(r'^\s*([a-zA-Z0-9_.\-]+)\s*:(?:\s*\d+)?\s+(.*)\s*$')
            # **************************

            for line in f:
                line_num += 1
                line_content = line.strip()
                if not line_content or line_content.startswith('#'): continue

                match = line_pattern.match(line_content)
                if match:
                    key = match.group(1)
                    # ***** Value part is now group 2 *****
                    value_part = match.group(2)
                    # *************************************
                    value = value_part # Default if no quotes

                    # Handle quotes and unescape internal quotes
                    if value_part.startswith('"') and value_part.endswith('"'):
                        value = value_part[1:-1].replace('\\"', '"')
                    elif value_part.startswith("'") and value_part.endswith("'"):
                         value = value_part[1:-1].replace("\\'", "'")

                    # Strip whitespace from the final value
                    inner_dict[key] = value.strip()
                else:
                    # This warning might still trigger for malformed lines, which is okay.
                    print(f"Warning: Skipping line {line_num} in {filepath} - does not match expected 'KEY: [index] \"Value\"' format: '{line_content}'", file=sys.stderr)

            return lang_key, inner_dict

    except FileNotFoundError:
        return None, None # Let caller handle missing file
    except ValueError as e:
        print(f"Error processing YAML data in {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading YAML {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None, None

    """
    Loads YAML data from a file.
    If is_vanilla_file is True, it's more lenient about the first line format.
    Returns lang_key (or None), data_dictionary (or None on error).
    Strips leading/trailing whitespace from values. <--- Added Note
    """
    lang_key = None
    inner_dict = {}
    line_num = 0
    try:
        # Use utf-8-sig to handle potential BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            line_num += 1
            if not first_line:
                print(f"Warning: File is empty: {filepath}", file=sys.stderr)
                return None, {}

            # Handle potential BOM if not handled by encoding='utf-8-sig' already
            # (utf-8-sig should handle it, but belt-and-suspenders)
            if first_line.startswith('\ufeff'):
                bom_stripped_line = first_line.lstrip('\ufeff').strip()
            else:
                bom_stripped_line = first_line.strip()

            match = re.match(r'^\s*(l_[a-zA-Z_]+)\s*:?\s*$', bom_stripped_line)
            if match:
                lang_key = match.group(1)
            else:
                f.seek(0) # Rewind to parse first line as data
                line_num = 0 # Reset line number for parsing
                if not is_vanilla_file:
                     print(f"Warning: First line in {filepath} ('{bom_stripped_line}') does not match 'l_language:'. Parsing values directly.", file=sys.stderr)
                lang_key = "l_unknown" # Assign placeholder

            line_pattern = re.compile(r'^\s*([a-zA-Z0-9_.\-]+)\s*:\s*(\d+)\s+(.*)\s*$')
            for line in f:
                line_num += 1
                line_content = line.strip()
                if not line_content or line_content.startswith('#'): continue

                match = line_pattern.match(line_content)
                if match:
                    key = match.group(1)
                    value_part = match.group(3) # Get the part after the index
                    value = value_part # Default if no quotes

                    # Handle quotes and unescape internal quotes
                    if value_part.startswith('"') and value_part.endswith('"'):
                        # Unescape quotes *within* the value first
                        value = value_part[1:-1].replace('\\"', '"')
                    elif value_part.startswith("'") and value_part.endswith("'"):
                         # Handle single quotes if necessary, though less common in HOI4
                         value = value_part[1:-1].replace("\\'", "'")

                    # ***** THE FIX: Strip whitespace from the final value *****
                    inner_dict[key] = value.strip()
                    # **********************************************************
                else:
                    print(f"Warning: Skipping line {line_num} in {filepath} - does not match expected 'KEY:index \"Value\"' format: '{line_content}'", file=sys.stderr)

            return lang_key, inner_dict

    except FileNotFoundError:
        # Let caller handle missing file, return None
        return None, None
    except ValueError as e:
        print(f"Error processing YAML data in {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading YAML {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None, None

    """
    Loads YAML data from a file.
    If is_vanilla_file is True, it's more lenient about the first line format.
    Returns lang_key (or None), data_dictionary (or None on error).
    """
    lang_key = None
    inner_dict = {}
    line_num = 0
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            line_num += 1
            if not first_line:
                print(f"Warning: File is empty: {filepath}", file=sys.stderr)
                # Return detected lang_key (None here) and empty dict
                return None, {}

            if first_line.startswith('\ufeff'):
                bom_stripped_line = first_line.lstrip('\ufeff').strip()
            else:
                bom_stripped_line = first_line.strip()

            # Match the language key line (e.g., l_english:)
            match = re.match(r'^\s*(l_[a-zA-Z_]+)\s*:?\s*$', bom_stripped_line)
            if match:
                lang_key = match.group(1)
            else:
                # If it doesn't match, rewind and parse from the start
                f.seek(0)
                # If it's expected to be a vanilla file, we might not *need* the key
                if not is_vanilla_file:
                     print(f"Warning: First line in {filepath} ('{bom_stripped_line}') does not match 'l_language:'. Parsing values directly.", file=sys.stderr)
                # Assign a placeholder key if needed, though it's often ignored for vanilla lookups
                lang_key = "l_unknown" # Placeholder

            # Parse subsequent lines
            line_pattern = re.compile(r'^\s*([a-zA-Z0-9_.\-]+)\s*:\s*(\d+)\s+(.*)\s*$')
            for line in f:
                line_num += 1
                line_content = line.strip()
                if not line_content or line_content.startswith('#'): continue
                match = line_pattern.match(line_content)
                if match:
                    key = match.group(1)
                    value_part = match.group(3)
                    # Handle quotes and unescape internal quotes
                    value = value_part
                    if value_part.startswith('"') and value_part.endswith('"'):
                        value = value_part[1:-1].replace('\\"', '"')
                    elif value_part.startswith("'") and value_part.endswith("'"):
                         value = value_part[1:-1].replace("\\'", "'")
                    inner_dict[key] = value
                else:
                    # Be less noisy for vanilla files, maybe just log once?
                    # For now, keep the warning
                    print(f"Warning: Skipping line {line_num} in {filepath} - does not match expected 'KEY:index \"Value\"' format: '{line_content}'", file=sys.stderr)

            return lang_key, inner_dict

    except FileNotFoundError:
        # Don't print error here, let the caller handle it (it's optional for vanilla)
        return None, None
    except ValueError as e: # Specific parsing errors
        print(f"Error processing YAML data in {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        return None, None
    except Exception as e: # Catch-all for other issues
        print(f"An unexpected error occurred loading YAML {filepath} (line ~{line_num}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None, None

# Map language codes (like 'en', 'ru') to HOI4 localisation folder names
# (This might need adjustments based on exact HOI4 folder names)
LANG_CODE_TO_FOLDER_NAME = {
    'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish',
    'pl': 'polish', 'pt': 'braz_por', 'ru': 'russian', 'ja': 'japanese',
    'zh': 'simp_chinese', 'ko': 'korean', 'it': 'italian', 'tr': 'turkish',
    # Add more as needed
}

# Map language codes to the conventional 'l_language' key used in files
LANG_CODE_TO_HOI4_KEY = {
    'en': 'l_english', 'de': 'l_german', 'fr': 'l_french', 'es': 'l_spanish',
    'pl': 'l_polish', 'pt': 'l_braz_por', 'ru': 'l_russian', 'ja': 'l_japanese',
    'zh': 'l_simp_chinese', 'ko': 'l_korean', 'it': 'l_italian', 'tr': 'l_turkish'
}


def write_yaml(filepath, lang_key, data):
    # ... (no changes needed)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{lang_key}:\n")
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    escaped_value = cleaned_value.replace('"', '\\"')
                    f.write(f' {key}:0 "{escaped_value}"\n')
                else:
                    f.write(f' {key}:0 {value}\n')
        return True
    except IOError as e:
        print(f"Error writing initial YAML file {filepath}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred writing initial YAML {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

def post_process_output_file(filepath, update_callback=None):
    # ... (no changes needed)
    try:
        # ... (rest of the function is identical)
        if not os.path.exists(filepath):
             if update_callback: update_callback(f"Error: File not found for post-processing: {filepath}")
             return False
        if update_callback: update_callback(f"Post-processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        original_content = content
        changed = False
        lines = content.splitlines()
        processed_lines = []
        line_pattern = re.compile(r'^(\s*[a-zA-Z0-9_.\-]+\s*:\s*\d+\s+)"(.*)"\s*$')
        log_once = True
        for i, line in enumerate(lines):
            if i == 0:
                processed_lines.append(line)
                continue
            match = line_pattern.match(line)
            if not match:
                processed_lines.append(line)
                continue
            key_part = match.group(1)
            value_part_escaped = match.group(2)
            value_part = value_part_escaped.replace('\\"', '"')
            cleaned_value = value_part
            line_changed = False
            if cleaned_value.startswith(': "') and cleaned_value.endswith('"'):
                cleaned_value = cleaned_value[3:-1]; line_changed = True
            if cleaned_value.startswith('"') and cleaned_value.endswith('"'):
                cleaned_value = cleaned_value[1:-1]; line_changed = True
            stripped_value = cleaned_value.strip()
            if stripped_value != cleaned_value:
                 cleaned_value = stripped_value; line_changed = True
            if line_changed: changed = True
            final_value_escaped = cleaned_value.replace('"', '\\"')
            processed_line = f'{key_part}"{final_value_escaped}"'
            if line_changed and update_callback and log_once:
                 update_callback(f"Post-processing: Cleaned formatting in one or more lines.")
                 log_once = False # Avoid spamming the log
            processed_lines.append(processed_line)
        if changed:
            new_content = "\n".join(processed_lines)
            if original_content.endswith('\n'): new_content += '\n'
            if update_callback: update_callback(f"Writing cleaned content back to {filepath}...")
            try:
                with open(filepath, 'w', encoding='utf-8') as f: f.write(new_content)
                if update_callback: update_callback("Post-processing finished successfully.")
            except IOError as e:
                 print(f"Error writing cleaned file {filepath}: {e}", file=sys.stderr)
                 if update_callback: update_callback(f"Error writing cleaned file: {e}")
                 return False
        else:
            if update_callback: update_callback("No post-processing changes needed for file content.")
        return True
    except Exception as e:
        print(f"Error during post-processing of {filepath}: {e}", file=sys.stderr)
        if update_callback: update_callback(f"Error during post-processing: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def preprocess_text(text, glossary):
    # ... (no changes needed)
    if not isinstance(text, str) or not text.strip(): return text, {}, {}
    # ... (rest of the function)
    placeholders = {}
    placeholder_token_prefix = "__PLACEHOLDER_"
    placeholder_counter = 0
    placeholder_regex = r'(\$[a-zA-Z0-9_]+\$|%[a-zA-Z%]|§[a-zA-Z0-9!RGBHUIY])'
    def replace_placeholder(match):
        nonlocal placeholder_counter
        original_placeholder = match.group(0)
        token = f"{placeholder_token_prefix}{placeholder_counter}__"
        placeholders[token] = original_placeholder
        placeholder_counter += 1
        return token
    processed_text = re.sub(placeholder_regex, replace_placeholder, text)
    glossary_tokens = {}
    glossary_token_prefix = "__GLOSSARY_"
    glossary_counter = 0
    if glossary:
        all_matches = []
        for source_term, target_term in glossary.items():
            if not source_term: continue
            try:
                term_regex = re.compile(r'\b' + re.escape(source_term) + r'\b', re.IGNORECASE)
                for match in term_regex.finditer(processed_text):
                    if not is_inside_quotes(processed_text, match.start()):
                        all_matches.append({ "start": match.start(), "end": match.end(), "target": target_term, "length": len(source_term) })
            except re.error as e:
                print(f"Warning: Skipping glossary term '{source_term}' due to regex error: {e}", file=sys.stderr)
                continue
        if all_matches:
            all_matches.sort(key=lambda m: (-m["length"], m["start"]))
            filtered_matches = []
            covered_indices = set()
            for match in all_matches:
                if any(i in covered_indices for i in range(match["start"], match["end"])): continue
                filtered_matches.append(match)
                for i in range(match["start"], match["end"]): covered_indices.add(i)
            filtered_matches.sort(key=lambda m: m["start"])
            current_pos = 0
            segments = []
            for match in filtered_matches:
                segments.append(processed_text[current_pos:match["start"]])
                token = f"{glossary_token_prefix}{glossary_counter}__"
                glossary_tokens[token] = match["target"]
                segments.append(token)
                glossary_counter += 1
                current_pos = match["end"]
            segments.append(processed_text[current_pos:])
            processed_text = "".join(segments)
    return processed_text, placeholders, glossary_tokens


def restore_text(translated_text, placeholders, glossary_tokens):
    # ... (no changes needed)
    for token, target_term in glossary_tokens.items():
        translated_text = translated_text.replace(token, target_term)
    for token, original_placeholder in placeholders.items():
        translated_text = translated_text.replace(token, original_placeholder)
    return translated_text

def translate_batch(texts, model, tokenizer, device):
    # ... (no changes needed)
    if not texts: return []
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_batch
    except Exception as e:
        print(f"Error during batch translation: {e}", file=sys.stderr)
        print(f"Problematic batch (first 5 items): {texts[:5]}", file=sys.stderr)
        return texts # Return originals on error


# --- Translation Runner (Modified) ---

def run_translation_logic(input_file, output_file, source_lang, target_lang, model_name,
                           glossary_file, batch_size=16,
                           hoi4_loc_folder=None, # NEW: Path to HOI4 'localisation' folder
                           update_callback=None):
    """Main logic including loading, checking vanilla, batch translation, writing, and post-processing."""
    if update_callback: update_callback("Starting translation process...")

    # 1. Load Glossary
    if update_callback: update_callback("Loading glossary...")
    glossary = load_glossary(glossary_file)

    # 2. Load Vanilla Data (if HOI4 loc folder provided)
    vanilla_english_data = None
    vanilla_target_data = None
    vanilla_files_loaded = False
    if hoi4_loc_folder and os.path.isdir(hoi4_loc_folder):
        if update_callback: update_callback(f"Attempting to load vanilla data from: {hoi4_loc_folder}")
        base_filename = os.path.basename(input_file)
        target_lang_filename = base_filename # Default to original if replacement fails
        source_lang_hoi4_key = LANG_CODE_TO_HOI4_KEY.get(source_lang)
        target_lang_hoi4_key = LANG_CODE_TO_HOI4_KEY.get(target_lang)
 
        if source_lang_hoi4_key and target_lang_hoi4_key and source_lang_hoi4_key in base_filename:
            target_lang_filename = base_filename.replace(source_lang_hoi4_key, target_lang_hoi4_key)
 

        english_folder = LANG_CODE_TO_FOLDER_NAME.get('en')
        target_folder = LANG_CODE_TO_FOLDER_NAME.get(target_lang)

        if english_folder and target_folder:
            vanilla_english_path = os.path.join(hoi4_loc_folder, english_folder, base_filename)
            vanilla_target_path = os.path.join(hoi4_loc_folder, target_folder, target_lang_filename)

            if update_callback: update_callback(f" -> Checking for vanilla English file: {vanilla_english_path}")
            _, vanilla_english_data = load_yaml(vanilla_english_path, is_vanilla_file=True)
            if vanilla_english_data is not None:
                if update_callback: update_callback(f"    -> Loaded {len(vanilla_english_data)} English vanilla entries.")
            else:
                 if update_callback: update_callback(f"    -> Vanilla English file not found or failed to load.")

            if update_callback: update_callback(f" -> Checking for vanilla Target file: {vanilla_target_path}")
            _, vanilla_target_data = load_yaml(vanilla_target_path, is_vanilla_file=True)
            if vanilla_target_data is not None:
                 if update_callback: update_callback(f"    -> Loaded {len(vanilla_target_data)} Target ({target_lang}) vanilla entries.")
            else:
                 if update_callback: update_callback(f"    -> Vanilla Target file not found or failed to load.")

            if vanilla_english_data is not None and vanilla_target_data is not None:
                vanilla_files_loaded = True
                if update_callback: update_callback(" -> Vanilla English and Target files loaded successfully for comparison.")
            else:
                 if update_callback: update_callback(" -> Could not load both required vanilla files for comparison. Will proceed without vanilla matching.")
        else:
            if update_callback: update_callback(f" -> Could not determine vanilla folder names for 'en' ({english_folder}) or '{target_lang}' ({target_folder}). Skipping vanilla check.")
    elif hoi4_loc_folder:
        if update_callback: update_callback(f"Warning: Provided HOI4 localisation folder path is not a valid directory: {hoi4_loc_folder}")


    # 3. Load Input YAML (Source Language)
    if update_callback: update_callback(f"Loading input source YAML: {input_file}")
    source_lang_key_detected, input_data = load_yaml(input_file)
    if input_data is None:
        if update_callback: update_callback("Failed to load input source YAML. Aborting.")
        return False
    total_keys_in_source = len(input_data)
    if update_callback: update_callback(f"Found {total_keys_in_source} keys in the source file.")

    # 4. Load Translation Model and Tokenizer (deferred)
    model = None
    tokenizer = None
    device = None
    model_loaded = False

    # 5. Prepare for Translation / Use Vanilla Match / Skip
    if update_callback: update_callback("Processing keys: Checking vanilla matches and preparing for AI...")
    items_to_translate = [] # List of tuples: (key, source_value) for AI
    translated_data = {} # Final output dictionary {key: target_value}
    skipped_vanilla_match = 0
    skipped_other = 0
    processed_keys = set()

    for key, user_source_value in input_data.items():
        if key in processed_keys: continue

        # Check for Vanilla Match first
        vanilla_match_found = False
        if vanilla_files_loaded:
            if key in vanilla_english_data and key in vanilla_target_data:
                # *** THE CORE CHECK ***
                if user_source_value == vanilla_english_data[key]:
                    translated_data[key] = vanilla_target_data[key] # Copy target translation
                    skipped_vanilla_match += 1
                    vanilla_match_found = True

        # If no vanilla match, decide if AI translation is needed
        if not vanilla_match_found:
            if isinstance(user_source_value, str) and user_source_value.strip():
                # Add non-empty strings to the list for AI translation
                items_to_translate.append((key, user_source_value))
            else:
                # Keep non-string or empty values as is
                translated_data[key] = user_source_value
                skipped_other += 1

        processed_keys.add(key)


    if update_callback:
        update_callback(f"Skipped {skipped_vanilla_match} keys (used direct vanilla translation).")
        update_callback(f"Skipped {skipped_other} keys (non-translatable: empty, non-string, etc.).")
        update_callback(f"Need to translate {len(items_to_translate)} keys using the AI model.")

    # 6. Perform AI Translation (if needed)
    if items_to_translate:
        if not model_loaded:
            # Load the model now
            if update_callback: update_callback(f"Loading model '{model_name}'...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if update_callback: update_callback(f"Using device: {device}")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                model.eval()
                model_loaded = True
                if update_callback: update_callback("Model and tokenizer loaded.")
            except Exception as e:
                # Handle model loading error
                error_msg = f"Error loading model or tokenizer '{model_name}': {e}"
                print(error_msg, file=sys.stderr)
                if update_callback: update_callback(error_msg)
                if "Helsinki-NLP" not in model_name:
                     suggested_model = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                     if update_callback: update_callback(f"Suggestion: Try model '{suggested_model}'.")
                return False # Cannot proceed

        # Preprocess items for AI
        if update_callback: update_callback("Preprocessing text for AI translation...")
        preprocessed_items = [(key, *preprocess_text(val, glossary)) for key, val in items_to_translate]

        # Batch Translation
        total_ai_items = len(preprocessed_items)
        ai_processed_count = 0
        if update_callback: update_callback(f"Starting AI translation in batches of {batch_size}...")
        num_batches = math.ceil(total_ai_items / batch_size)

        for i in range(0, total_ai_items, batch_size):
            batch_slice = preprocessed_items[i:min(i + batch_size, total_ai_items)]
            current_batch_num = (i // batch_size) + 1
            if not batch_slice: continue

            keys_in_batch = [item[0] for item in batch_slice]
            texts_in_batch = [item[1] for item in batch_slice]
            placeholders_in_batch = [item[2] for item in batch_slice]
            glossary_tokens_in_batch = [item[3] for item in batch_slice]

            if update_callback: update_callback(f"Translating batch {current_batch_num}/{num_batches} ({len(texts_in_batch)} items)...")
            translated_batch = translate_batch(texts_in_batch, model, tokenizer, device)

            # Process batch results
            if len(translated_batch) != len(keys_in_batch):
                 if update_callback: update_callback(f"ERROR: Batch size mismatch! Input: {len(keys_in_batch)}, Output: {len(translated_batch)}. Marking affected keys.")
                 for k in keys_in_batch:
                     if k not in translated_data: translated_data[k] = f"ERROR: BATCH_TRANSLATION_FAILED"
                 ai_processed_count += len(keys_in_batch)
                 continue

            for j, translated_text in enumerate(translated_batch):
                key = keys_in_batch[j]
                placeholders = placeholders_in_batch[j]
                glossary_tokens = glossary_tokens_in_batch[j]
                final_text = restore_text(translated_text, placeholders, glossary_tokens)
                if key not in translated_data: # Add AI result to final dict
                     translated_data[key] = final_text
                ai_processed_count += 1

            if update_callback: update_callback(f"Finished batch {current_batch_num}/{num_batches}. AI processed: {ai_processed_count}/{total_ai_items}")

        if update_callback: update_callback(f"Finished AI translation for {ai_processed_count} keys.")
    else:
        if update_callback: update_callback("No items required AI translation.")


    # 7. Determine output language key
    # Use the mapping from code to key (e.g., ru -> l_russian)
    output_lang_key = LANG_CODE_TO_HOI4_KEY.get(target_lang, f"l_{target_lang}") # Fallback to l_code
    if update_callback: update_callback(f"Using output language key: {output_lang_key}")

    # 8. Write Output YAML
    if update_callback: update_callback("Writing final YAML file...")
    # Ensure all original keys are present in the final output
    final_output_data = {}
    missing_keys_count = 0
    for key in input_data:
        if key in translated_data:
            final_output_data[key] = translated_data[key]
        else:
            # This indicates a logic error somewhere above if reached
            final_output_data[key] = f"ERROR: KEY_MISSING_AT_WRITE"
            missing_keys_count += 1
            if update_callback: update_callback(f"Critical Warning: Key '{key}' was missing from final data before writing!")

    if missing_keys_count > 0 and update_callback:
        update_callback(f"Critical Warning: {missing_keys_count} keys were unexpectedly missing!")

    write_success = write_yaml(output_file, output_lang_key, final_output_data)
    if not write_success:
         if update_callback: update_callback("Error: Failed to write final output file. Aborting.")
         return False

    # 9. Post-Processing Step
    post_process_success = post_process_output_file(output_file, update_callback)
    if not post_process_success:
        if update_callback: update_callback("Warning: Post-processing step encountered errors. File might contain artifacts.")

    # Final Success Message
    total_output_keys = len(final_output_data)
    ai_translated_count = len(items_to_translate) # Count how many *should* have been AI translated
    if update_callback:
        update_callback(f"Translation process finished!")
        update_callback(f" - {skipped_vanilla_match} keys used direct vanilla translation.")
        update_callback(f" - {ai_translated_count} keys processed by AI (or attempted).")
        update_callback(f" - {skipped_other} keys skipped (non-translatable).")
        update_callback(f" - Total keys in output: {total_output_keys} (expected: {total_keys_in_source}).")
        update_callback(f"Output saved to {output_file}")
    return True


# --- GUI Implementation (Modified) ---

class TranslationGUI:
    def __init__(self, master):
        self.master = master
        master.title("HOI4 Localization Translator")
        master.geometry("700x620") # Adjusted height
        self.style = ttk.Style()
        try: self.style.theme_use('clam')
        except tk.TclError: self.style.theme_use('default')

        self.frame = ttk.Frame(master, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # --- Widgets ---
        row_idx = 0
        ttk.Label(self.frame, text="Input File (.yml, Source Lang):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.input_file_entry = ttk.Entry(self.frame, width=50)
        self.input_file_entry.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse...", command=self.browse_input_file).grid(row=row_idx, column=2, padx=5)
        row_idx += 1

        ttk.Label(self.frame, text="Output File (.yml, Target Lang):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.output_file_entry = ttk.Entry(self.frame, width=50)
        self.output_file_entry.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse...", command=self.browse_output_file).grid(row=row_idx, column=2, padx=5)
        row_idx += 1

        # --- HOI4 Loc Folder ---
        ttk.Label(self.frame, text="HOI4 Loc Folder (Optional):").grid(row=row_idx, column=0, sticky=tk.W, pady=(10, 2))
        self.hoi4_loc_folder_entry = ttk.Entry(self.frame, width=50)
        self.hoi4_loc_folder_entry.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=(10, 2))
        ttk.Button(self.frame, text="Browse...", command=self.browse_hoi4_loc_folder).grid(row=row_idx, column=2, padx=5, pady=(10, 2))
        row_idx += 1
        # --- End HOI4 Loc Folder ---

        ttk.Label(self.frame, text="Source Lang (code):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.source_lang_entry = ttk.Entry(self.frame, width=10)
        self.source_lang_entry.grid(row=row_idx, column=1, sticky=tk.W, pady=2)
        self.source_lang_entry.insert(0, "en")
        row_idx += 1

        ttk.Label(self.frame, text="Target Lang (code):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.target_lang_entry = ttk.Entry(self.frame, width=10)
        self.target_lang_entry.grid(row=row_idx, column=1, sticky=tk.W, pady=2)
        self.target_lang_entry.insert(0, "ru")
        row_idx += 1

        ttk.Label(self.frame, text="Translation Model:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.model_name_entry = ttk.Entry(self.frame, width=50)
        self.model_name_entry.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=2)
        # Update default model suggestion logic might need adjustment if source/target changes
        self.source_lang_entry.bind("<FocusOut>", self.update_default_model)
        self.target_lang_entry.bind("<FocusOut>", self.update_default_model)
        self.target_lang_entry.bind("<Return>", self.update_default_model)
        self.update_default_model() # Call once initially
        row_idx += 1


        ttk.Label(self.frame, text="Batch Size:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.batch_size_entry = ttk.Entry(self.frame, width=10)
        self.batch_size_entry.grid(row=row_idx, column=1, sticky=tk.W, pady=2)
        self.batch_size_entry.insert(0, "16")
        row_idx += 1


        ttk.Label(self.frame, text="Glossary File (.csv, Optional):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.glossary_file_entry = ttk.Entry(self.frame, width=50)
        self.glossary_file_entry.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.frame, text="Browse...", command=self.browse_glossary_file).grid(row=row_idx, column=2, padx=5)
        row_idx += 1

        self.translate_button = ttk.Button(self.frame, text="Translate", command=self.start_translation_thread)
        self.translate_button.grid(row=row_idx, column=0, columnspan=3, pady=15)
        row_idx += 1

        ttk.Label(self.frame, text="Status Log:").grid(row=row_idx, column=0, sticky=tk.W, pady=(10, 0))
        row_idx += 1
        self.log_area = scrolledtext.ScrolledText(self.frame, height=10, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.log_area.grid(row=row_idx, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(row_idx, weight=1)

        self.log_message("Ready. Fill in details.")
        self.log_message("Optional: Provide HOI4 'localisation' folder path to reuse exact vanilla translations.")
        self.log_message("Requires: transformers, torch, PyYAML")

    # --- Browse Methods ---
    def browse_input_file(self):
        filepath = filedialog.askopenfilename(title="Select Input Localization File (Source Language)", filetypes=(("YAML files", "*.yml"),("All files", "*.*")))
        if filepath:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, filepath)
            self.suggest_output_filename(filepath) # Suggest output based on new input

    def browse_output_file(self):
        # ... (logic seems fine, maybe simplify default name generation slightly)
        default_name = ""
        input_path = self.input_file_entry.get()
        target_lang = self.target_lang_entry.get().strip()
        if input_path and target_lang:
            base, ext = os.path.splitext(os.path.basename(input_path))
            # Try to replace source lang part if possible
            source_lang_hoi4 = LANG_CODE_TO_HOI4_KEY.get(self.source_lang_entry.get().strip(), f"l_{self.source_lang_entry.get().strip()}")
            target_lang_hoi4 = LANG_CODE_TO_HOI4_KEY.get(target_lang, f"l_{target_lang}")
            if source_lang_hoi4 in base:
                new_base = base.replace(source_lang_hoi4, target_lang_hoi4)
            else: # Fallback: Append target lang key
                new_base = f"{base}_{target_lang_hoi4}"
            default_name = f"{new_base}{ext}"
        elif input_path: default_name = os.path.basename(input_path)

        filepath = filedialog.asksaveasfilename(title="Select Output File Location", filetypes=(("YAML files", "*.yml"),("All files", "*.*")), defaultextension=".yml", initialfile=default_name)
        if filepath:
            self.output_file_entry.delete(0, tk.END)
            self.output_file_entry.insert(0, filepath)


    def browse_hoi4_loc_folder(self): # New browse method
        directory = filedialog.askdirectory(title="Select HOI4 'localisation' Folder (Optional)")
        if directory:
             # Basic check if it looks like a HOI4 loc folder
             if os.path.exists(os.path.join(directory, "english")):
                 self.hoi4_loc_folder_entry.delete(0, tk.END)
                 self.hoi4_loc_folder_entry.insert(0, directory)
             else:
                 messagebox.showwarning("Warning", f"Selected folder '{directory}' doesn't seem to contain an 'english' subfolder. Please select the main 'localisation' directory.")


    def browse_glossary_file(self):
        filepath = filedialog.askopenfilename(title="Select Glossary File (Optional)", filetypes=(("CSV files", "*.csv"),("All files", "*.*")))
        if filepath:
            self.glossary_file_entry.delete(0, tk.END)
            self.glossary_file_entry.insert(0, filepath)

    # --- Other GUI Methods ---
    def suggest_output_filename(self, input_filepath):
        try:
            target_lang = self.target_lang_entry.get().strip()
            source_lang = self.source_lang_entry.get().strip()
            if not target_lang or not source_lang or not input_filepath: return

            dirname = os.path.dirname(input_filepath)
            base, ext = os.path.splitext(os.path.basename(input_filepath))

            source_lang_hoi4 = LANG_CODE_TO_HOI4_KEY.get(source_lang, f"l_{source_lang}")
            target_lang_hoi4 = LANG_CODE_TO_HOI4_KEY.get(target_lang, f"l_{target_lang}")

            # Try replacing the source language key part in the filename
            new_base = base
            if source_lang_hoi4 in base:
                 new_base = base.replace(source_lang_hoi4, target_lang_hoi4)
            # If replacement didn't happen (e.g. source key not in filename), maybe append?
            # This part is tricky, depends on user naming conventions. Sticking to replacement is safer.
            elif f"_{source_lang}" in base: # Try replacing _en suffix etc.
                 new_base = base.replace(f"_{source_lang}", f"_{target_lang}")

            # Avoid suggesting the same name as input if langs are different
            if new_base == base and source_lang != target_lang:
                 new_base = f"{base}_{target_lang_hoi4}" # Fallback append if different langs

            suggested_output = os.path.join(dirname, f"{new_base}{ext}")

            # Only update if the suggestion is different from current output & input
            current_output = self.output_file_entry.get()
            if suggested_output != current_output and suggested_output != input_filepath:
                self.output_file_entry.delete(0, tk.END)
                self.output_file_entry.insert(0, suggested_output)
                # self.log_message(f"Suggested output: {suggested_output}") # Optional log
        except Exception as e:
            self.log_message(f"Could not auto-suggest output filename: {e}")


    def update_default_model(self, event=None):
        source_lang = self.source_lang_entry.get().strip()
        target_lang = self.target_lang_entry.get().strip()
        if source_lang and target_lang:
            model_suggestion = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            current_model = self.model_name_entry.get()
            # Update if empty or looks like a Helsinki model for *any* lang pair
            if not current_model or re.match(r"Helsinki-NLP/opus-mt-\w+-\w+", current_model):
                 if model_suggestion != current_model:
                    self.model_name_entry.delete(0, tk.END)
                    self.model_name_entry.insert(0, model_suggestion)
                    # Only log if model actually changes suggestion
                    # self.log_message(f"Suggested model: {model_suggestion}")
        # Also update output filename suggestion when language changes
        if self.input_file_entry.get():
            self.suggest_output_filename(self.input_file_entry.get())


    def log_message(self, message):
        # ... (no changes needed)
        def append_log():
            if self.log_area.winfo_exists():
                self.log_area.config(state=tk.NORMAL)
                self.log_area.insert(tk.END, message + "\n")
                self.log_area.see(tk.END)
                self.log_area.config(state=tk.DISABLED)
        try: self.master.after(0, append_log)
        except tk.TclError as e: print(f"Log (GUI Error: {e}): {message}", file=sys.stderr)


    def set_ui_state(self, enabled):
        """Enable/disable UI elements during processing."""
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets_to_toggle = [
            self.input_file_entry, self.output_file_entry,
            self.hoi4_loc_folder_entry, # New
            self.source_lang_entry, self.target_lang_entry,
            self.model_name_entry, self.glossary_file_entry,
            self.batch_size_entry,
            self.translate_button
        ]
        # Toggle associated Browse buttons
        for widget in self.frame.winfo_children():
             if isinstance(widget, ttk.Button) and "Browse" in widget.cget("text"):
                 widgets_to_toggle.append(widget)
        for widget in widgets_to_toggle:
             if widget.winfo_exists():
                 try: widget.config(state=state)
                 except tk.TclError: pass

    def start_translation_thread(self):
        """Gets parameters from GUI and starts translation in a new thread."""
        input_file = self.input_file_entry.get().strip()
        output_file = self.output_file_entry.get().strip()
        source_lang = self.source_lang_entry.get().strip()
        target_lang = self.target_lang_entry.get().strip()
        model_name = self.model_name_entry.get().strip()
        glossary_file = self.glossary_file_entry.get().strip() or None
        hoi4_loc_folder = self.hoi4_loc_folder_entry.get().strip() or None # New

        try:
            batch_size = int(self.batch_size_entry.get().strip())
            if batch_size <= 0: raise ValueError("Batch size must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid Batch Size: {e}")
            return

        # --- Validation ---
        required_fields = {"Input File": input_file, "Output File": output_file, "Source Lang": source_lang, "Target Lang": target_lang, "Model Name": model_name}
        missing = [name for name, value in required_fields.items() if not value]
        if missing: messagebox.showerror("Error", f"Please fill in required fields: {', '.join(missing)}"); return

        if not os.path.exists(input_file): messagebox.showerror("Error", f"Input file not found:\n{input_file}"); return
        if glossary_file and not os.path.exists(glossary_file): messagebox.showwarning("Warning", f"Glossary file not found (will be ignored):\n{glossary_file}"); glossary_file = None
        if hoi4_loc_folder and not os.path.isdir(hoi4_loc_folder): messagebox.showwarning("Warning", f"HOI4 Loc Folder path is not a valid directory (will be ignored):\n{hoi4_loc_folder}"); hoi4_loc_folder = None

        # Check if target lang has a mapping for vanilla folder
        if hoi4_loc_folder and not LANG_CODE_TO_FOLDER_NAME.get(target_lang):
            messagebox.showwarning("Warning", f"Target language '{target_lang}' doesn't have a known HOI4 folder mapping. Cannot perform vanilla comparison for this language.")
            hoi4_loc_folder = None # Disable vanilla check

        # Lang code format check (optional)
        # ... (keep previous lang code checks if desired)

        # --- Start Thread ---
        self.set_ui_state(False)
        self.log_message("="*20 + "\nStarting New Translation" + "\n" + "="*20)
        # ... (Log other parameters)
        self.log_message(f"HOI4 Loc Folder: {hoi4_loc_folder if hoi4_loc_folder else 'None (Vanilla check disabled)'}")
        self.log_message("Starting translation thread...")

        thread = threading.Thread(
            target=self.run_translation_in_thread,
            args=(input_file, output_file, source_lang, target_lang, model_name, glossary_file, batch_size, hoi4_loc_folder), # Pass new arg
            daemon=True
        )
        thread.start()

    def run_translation_in_thread(self, input_file, output_file, source_lang, target_lang, model_name, glossary_file, batch_size, hoi4_loc_folder): # Updated args
        """Wrapper function to run the logic and handle results/errors from the thread."""
        try:
            success = run_translation_logic(
                input_file, output_file, source_lang, target_lang, model_name,
                glossary_file, batch_size,
                hoi4_loc_folder, # Pass new arg
                update_callback=self.log_message
            )
            # Schedule GUI updates back on the main thread
            if success: self.master.after(0, lambda: messagebox.showinfo("Success", f"Translation completed successfully!\nOutput saved to:\n{output_file}"))
            else: self.master.after(0, lambda: messagebox.showerror("Error", "Translation process failed or finished with warnings. Check the log for details."))
        except Exception as e:
            # ... (error handling)
            error_msg = f"An unexpected fatal error occurred in the translation thread: {e}"
            print(error_msg, file=sys.stderr); import traceback; traceback.print_exc(file=sys.stderr)
            self.log_message(f"FATAL ERROR: {error_msg}\nTraceback:\n{traceback.format_exc()}")
            self.master.after(0, lambda: messagebox.showerror("Fatal Error", f"{error_msg}\n\nSee log or console for technical details."))
        finally:
            # Ensure UI is re-enabled
            self.master.after(0, lambda: self.set_ui_state(True))


# --- Main Execution (Modified) ---

def main_cli():
    parser = argparse.ArgumentParser(
        description="Translate HOI4 localization YAML files using Hugging Face models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument("input_file", help="Path to the input localization YAML file (source language).")
    parser.add_argument("output_file", help="Path to save the translated YAML file (target language).")
    parser.add_argument("source_lang", help="Source language code (e.g., 'en').")
    parser.add_argument("target_lang", help="Target language code (e.g., 'ru').")
    parser.add_argument("model_name", help="Name of the Hugging Face translation model (e.g., 'Helsinki-NLP/opus-mt-en-ru').")
    # Optional arguments
    parser.add_argument("--glossary", help="Optional path to a glossary CSV file.", default=None)
    parser.add_argument("--batch_size", type=int, default=16, help="Number of strings to translate per batch.")
    parser.add_argument("--hoi4-loc-folder", help="Optional path to the main HOI4 'localisation' folder to enable vanilla matching.", default=None) # New CLI arg
    parser.add_argument("--gui", action="store_true", help="Launch the Graphical User Interface instead of CLI.")

    # Check for --gui early
    if "--gui" in sys.argv: print("Launching GUI mode..."); main_gui(); sys.exit()

    args = parser.parse_args()

    # Validate folder path if provided
    if args.hoi4_loc_folder and not os.path.isdir(args.hoi4_loc_folder):
        print(f"Error: Provided HOI4 localisation folder path is not a valid directory: {args.hoi4_loc_folder}", file=sys.stderr)
        sys.exit(1)
    # Validate target lang mapping for vanilla check
    if args.hoi4_loc_folder and not LANG_CODE_TO_FOLDER_NAME.get(args.target_lang):
         print(f"Warning: Target language '{args.target_lang}' lacks a folder mapping for vanilla check. Disabling vanilla matching.", file=sys.stderr)
         args.hoi4_loc_folder = None # Disable it

    def console_logger(message): print(message)

    print("Running in Command-Line Interface mode.")
    success = run_translation_logic(
        args.input_file, args.output_file, args.source_lang, args.target_lang, args.model_name,
        args.glossary, args.batch_size,
        args.hoi4_loc_folder,           # Pass new arg
        update_callback=console_logger
    )
    if not success: print("Translation process finished with errors.", file=sys.stderr); sys.exit(1)
    else: print("Translation process finished successfully.")

def main_gui():
    """Launches the GUI."""
    root = tk.Tk()
    gui = TranslationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Simplified startup logic
    if "--gui" in sys.argv or len(sys.argv) == 1:
         print("Launching GUI mode...")
         main_gui()
    else:
        main_cli()