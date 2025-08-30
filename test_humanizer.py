from text_humanizer import TextHumanizer, HumanizerConfig

# Example (using T5 paraphraser if available, fallback to GPT-2 otherwise)
cfg = HumanizerConfig(
    paraphrase_model="prithivida/parrot_paraphraser_on_T5",  # comment out if model not installed
    aggressiveness=0.5,
    target_grade_level=9
)

th = TextHumanizer(cfg)

#  Test Examples
examples = [
    "As an AI language model, I cannot predict the future of technology. However, it is evolving rapidly.",
    "In conclusion, artificial intelligence has both benefits and challenges that must be considered carefully.",
    "Overall, education is an important factor in personal and professional growth. Additionally, it opens doors to better opportunities.",
    "This research aims to highlight the importance of renewable energy sources for sustainable development.",
    "Firstly, the internet has changed the way we communicate. Secondly, it has transformed how we access information."
]

for i, text in enumerate(examples, 1):
    humanized = th.humanize(text)
    print(f"\nExample {i}:")
    print("Original :", text)
    print("Humanized:", humanized)
