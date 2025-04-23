A Python tool designed to help Hearts of Iron IV modders translate their localization `.yml` files efficiently using Hugging Face transformer models. It offers features like batch translation for speed, custom glossaries for term consistency, and the ability to automatically reuse existing vanilla game translations.

## Key Features

*   **AI Translation:** Leverages Hugging Face `transformers` library for sequence-to-sequence translation models (e.g., Helsinki-NLP).
*   **Batch Processing:** Translates multiple localization strings in batches for significant speed improvements, especially on GPUs.
*   **Glossary Support:** Use a custom CSV glossary (`SourceTerm,TargetTerm`) to ensure consistent translation of specific game terms, names, or phrases.
*   **Vanilla Translation Reuse:** Optionally compares source text values against vanilla English files; if an exact match is found, it uses the corresponding vanilla translation from the target language file, skipping AI translation for that entry.
*   **HOI4 YAML Parsing:** Specifically parses the `l_language:` format common in HOI4 mods, handling `KEY: "Value"` and `KEY:0 "Value"` styles.
*   **Placeholder Handling:** Preserves common HOI4 placeholders (`$VAR$`, `§C`, `%d`, etc.) during translation.
*   **Post-Processing:** Includes basic cleanup rules to fix common artifacts introduced by translation models (e.g., extra quotes, colons).
*   **Dual Interface:** Offers both a user-friendly Graphical User Interface (GUI) built with Tkinter and a Command-Line Interface (CLI) for automation.

## Requirements

*   Python 3.7+
*   PyTorch (`torch`)
*   Transformers (`transformers`)
*   PyYAML (`PyYAML`)
*   A CUDA-enabled GPU is **highly recommended** for reasonable translation speeds, especially with batching. CPU translation will be significantly slower.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/hoi4-localization-translator.git # Replace with your repo URL
    cd hoi4-localization-translator
    ```
2.  **Install required libraries:**
    ```bash
    pip install torch transformers PyYAML
    ```
    *(Note: Ensure you install the correct PyTorch version for your system, especially if using CUDA. See the [official PyTorch instructions](https://pytorch.org/get-started/locally/)).*

## Usage

### Graphical User Interface (GUI)

Launch the GUI by running the script without arguments or with the `--gui` flag:

```bash
python hoi4_translator.py
# or
python hoi4_translator.py --gui
