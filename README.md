Text Humanizer Tool

The Text Humanizer Tool is a Python utility that rewrites machine-like or robotic sentences into more natural, human-sounding text.
It uses NLP techniques such as paraphrasing (via T5 or GPT-2), synonym substitution (WordNet), readability adjustment, and rule-based cleanup to make text flow naturally while preserving meaning.

How It Works

Sentence Segmentation → Breaks text into sentences using spaCy or NLTK.

Debotification → Removes phrases like “As an AI language model”, “In conclusion”, etc.

Paraphrasing → Attempts to rephrase using:

A paraphrase model (e.g., T5 Parrot paraphraser) if available.

GPT-2 (causal LM generation) as fallback.

Synonym tweaking as a last resort.

Candidate Selection → Compares paraphrased sentences to the original and chooses the most natural version.

Polishing & Readability → Adjusts punctuation, merges short sentences, and adapts reading level to the target Flesch–Kincaid grade.

Final Output → Returns rewritten text with improved flow, less repetition, and a more natural style.

 Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/text-humanizer.git
cd text-humanizer

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt


Dependencies include:

torch

transformers

nltk

spacy

sacrebleu

rapidfuzz

textstat

(Ensure you also install the small English model for spaCy if not already installed:)

python -m spacy download en_core_web_sm

Usage
Command Line
python text_humanizer.py --text "As an AI language model, I cannot predict the future of technology."


With input file:

python text_humanizer.py --input-file input.txt --out output.txt


Available options:

--text             Direct input text
--input-file       Path to input text file
--out              Path to write humanized text
--model            Paraphrase model (e.g. prithivida/parrot_paraphraser_on_T5)
--aggressiveness   0..1 (higher = more rewriting)
--target-grade     Target readability grade (default=9)
--gpt2             GPT-2 model ID (default=gpt2)

 Example
Input
As an AI language model, I cannot predict the future of technology. However, it is evolving rapidly.

Output
I can't predict where technology will go, but it's advancing quickly.

Input
In conclusion, artificial intelligence has both benefits and challenges that must be considered carefully.

Output
To wrap up, artificial intelligence brings both advantages and difficulties that need careful thought.

 Notes

If the T5 paraphraser is unavailable, the tool falls back to GPT-2 or WordNet synonyms.

Randomness is seeded for reproducibility, but results may still vary slightly.

Works best on robotic, AI-like text and overly formal writing.