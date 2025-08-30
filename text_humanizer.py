from __future__ import annotations
import argparse
import os
import re
import torch
import sys
import random
from dataclasses import dataclass
from typing import List, Optional

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from rapidfuzz.distance import Levenshtein
import textstat
import sacrebleu
import spacy

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed,
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


@dataclass
class HumanizerConfig:
    paraphrase_model: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50
    num_return_sequences: int = 3
    seed: int = 42
    target_grade_level: int = 9
    preserve_numbers: bool = True
    preserve_entities: bool = True
    aggressiveness: float = 0.45
    min_improvement_bleu: float = 0.05
    max_semantic_drift: float = 0.25
    use_spacy_model: str = 'en_core_web_sm'
    use_gpt2: str = 'gpt2'


class TextHumanizer:
    def __init__(self, config: HumanizerConfig | None = None):
        self.cfg = config or HumanizerConfig()
        set_seed(self.cfg.seed)

        try:
            self.nlp = spacy.load(
                self.cfg.use_spacy_model,
                disable=["ner"] if not self.cfg.preserve_entities else []
            )
        except OSError:
            try:
                from spacy.cli import download as spacy_download
                spacy_download(self.cfg.use_spacy_model)
                self.nlp = spacy.load(self.cfg.use_spacy_model)
            except Exception:
                self.nlp = spacy.blank("en")
                self.nlp.add_pipe("sentencizer")

        self.paraphraser = None
        if self.cfg.paraphrase_model:
            try:
                self.paraphraser = pipeline("text2text-generation", model=self.cfg.paraphrase_model)
            except Exception:
                self.paraphraser = None

        try:
            self.gpt2_tok = AutoTokenizer.from_pretrained(self.cfg.use_gpt2)
            self.gpt2 = AutoModelForCausalLM.from_pretrained(self.cfg.use_gpt2)
        except Exception:
            self.gpt2_tok = None
            self.gpt2 = None

    def humanize(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text

        sents = self._segment_sentences(text)
        sents = [self._normalize_whitespace(s) for s in sents]
        meta = self._analyze(sents)

        sents = [self._debotify(s) for s in sents]
        rewritten = self._rewrite_pass(sents)
        polished = self._polish(rewritten, original_sents=sents)
        out = self._postprocess(polished)

        if not self._acceptable_change(" ".join(sents), out):
            old_aggr = self.cfg.aggressiveness
            self.cfg.aggressiveness = min(0.75, old_aggr + 0.2)
            try:
                alt = self._postprocess(
                    self._polish(self._rewrite_pass(sents), original_sents=sents)
                )
                if self._acceptable_change(" ".join(sents), alt):
                    out = alt
            finally:
                self.cfg.aggressiveness = old_aggr

        return out

    def _segment_sentences(self, text: str) -> List[str]:
        if self.nlp and ("senter" in self.nlp.pipe_names or
                         "sentencizer" in self.nlp.pipe_names or
                         len(self.nlp.pipe_names) > 0):
            doc = self.nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    @staticmethod
    def _normalize_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _analyze(self, sents: List[str]) -> dict:
        openers = [re.match(r"^(In conclusion|Overall|Additionally|Moreover|Furthermore|However|That said|As an AI|As a language model)\b", s, re.I)
                   for s in sents]
        opener_flags = [bool(m) for m in openers]
        mean_len = sum(len(s) for s in sents) / max(1, len(sents))
        long_sents = [i for i, s in enumerate(sents) if len(s.split()) > 38]
        repeats = self._repetitiveness_score(" ".join(sents))
        return {
            "opener_flags": opener_flags,
            "mean_len": mean_len,
            "long_sents": long_sents,
            "repeats": repeats,
        }

    def _debotify(self, s: str) -> str:
        patterns = [
            (r"\bAs an AI( language model)?[, ]?\b", ""),
            (r"\bI (cannot|can't) (.*?)(?:\.|$)", lambda m: ""),
            (r"\bIn conclusion,?\b", "To wrap up,"),
            (r"\bOverall,?\b", "All in all,"),
            (r"\bAdditionally,?\b", "Also,"),
            (r"\bMoreover,?\b", "Plus,"),
            (r"\bFurthermore,?\b", "What's more,"),
        ]
        out = s
        for pat, repl in patterns:
            out = re.sub(pat, repl if isinstance(repl, str) else repl, out, flags=re.I)
        out = re.sub(r"\(\s*\d+\s*\)|\b(Firstly|Secondly|Thirdly)\b", "", out, flags=re.I)
        return self._normalize_whitespace(out)

    def _rewrite_pass(self, sents: List[str]) -> List[List[str]]:
        candidates_per_sentence: List[List[str]] = []
        for s in sents:
            cands = self._paraphrase_sentence(s) or []
            cands.append(s)
            uniq = self._unique_by_edit_distance(cands)
            candidates_per_sentence.append(uniq[: max(1, self.cfg.num_return_sequences)])
        return candidates_per_sentence

    def _paraphrase_sentence(self, s: str) -> Optional[List[str]]:
        outputs: List[str] = []

        if self.paraphraser is not None:
            try:
                prompt = f"paraphrase: {s}"
                gens = self.paraphraser(
                    prompt,
                    max_new_tokens=min(self.cfg.max_length, len(s) + 60),
                    num_return_sequences=self.cfg.num_return_sequences,
                    temperature=self.cfg.temperature,
                    do_sample=True,
                    top_p=self.cfg.top_p
                )
                for g in gens:
                    txt = g["generated_text"] if isinstance(g, dict) else str(g)
                    txt = self._normalize_whitespace(txt.replace(",,", ","))
                    if txt and not txt.endswith("."):
                        txt += "."
                    outputs.append(txt)
            except Exception:
                pass

        if self.gpt2 is not None and self.gpt2_tok is not None and not outputs:
            try:
                prompt = self._gpt2_rewrite_prompt(s)
                input_ids = self.gpt2_tok.encode(prompt, return_tensors="pt")
                gen = self.gpt2.generate(
                    input_ids,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    num_return_sequences=self.cfg.num_return_sequences,
                    pad_token_id=self.gpt2_tok.eos_token_id,
                    eos_token_id=self.gpt2_tok.eos_token_id,
                    no_repeat_ngram_size=2
                )
                for out in gen:
                    txt = self.gpt2_tok.decode(out, skip_special_tokens=True)
                    txt = txt.replace(prompt, "").strip()
                    if txt and not txt.endswith("."):
                        txt += "."
                    outputs.append(self._normalize_whitespace(txt))
            except Exception:
                outputs.append(self._synonym_tweak(s))

        if not outputs:
            outputs.append(self._synonym_tweak(s))

        return [o for o in outputs if o]

    def _gpt2_rewrite_prompt(self, s: str) -> str:
        return (
            "Rewrite the following sentence to sound natural and conversational, "
            "keep the meaning the same, avoid robotic phrasing, and keep named entities intact.\n"
            f"Sentence: {s}\nRewritten: "
        )

    def _synonym_tweak(self, s: str) -> str:
        tokens = s.split()
        if len(tokens) < 5:
            return s
        idxs = list(range(1, len(tokens) - 1))
        random.shuffle(idxs)
        changes = 0
        for i in idxs:
            w = re.sub(r"\W", "", tokens[i])
            if not w or w[0].isupper():
                continue
            syns = {l.name().replace('_', ' ') for syn in wn.synsets(w) for l in syn.lemmas() if l.name().lower() != w.lower()}
            syns = [x for x in syns if 3 <= len(x) <= 12 and ' ' not in x]
            if syns:
                tokens[i] = tokens[i].replace(w, random.choice(syns))
                changes += 1
            if changes >= 2 * self.cfg.aggressiveness:
                break
        return self._normalize_whitespace(" ".join(tokens))

    def _unique_by_edit_distance(self, cands: List[str], threshold: int = 4) -> List[str]:
        uniq = []
        for c in cands:
            if all(Levenshtein.distance(c, u) > threshold for u in uniq):
                uniq.append(c)
        return uniq

    def _polish(self, candidates_per_sentence: List[List[str]], original_sents: List[str]) -> List[str]:
        polished = []
        for cands, orig in zip(candidates_per_sentence, original_sents):
            best = self._choose_best_candidate(cands, orig)
            best = self._tidy_sentence(best)
            polished.append(best)
        polished = self._join_for_flow(polished)
        polished = [self._adjust_readability(s) for s in polished]
        return polished

    def _choose_best_candidate(self, cands: List[str], original: str) -> str:
        def simple_similarity(a: str, b: str) -> float:
            ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        best, best_score = None, -1e9
        for c in cands:
            sim = simple_similarity(c, original)
            robo = self._robotic_penalty(c)
            score = 2.0 * sim - robo
            if score > best_score:
                best, best_score = c, score
        return best or original

    def _robotic_penalty(self, s: str) -> float:
        penalty = 0.0
        penalty += 0.3 if re.search(r"\bIn conclusion\b|\bAs an AI\b|\bOverall\b", s, re.I) else 0.0
        penalty += 0.2 if re.search(r"\bFirstly\b|\bSecondly\b|\bThirdly\b", s, re.I) else 0.0
        penalty += 0.2 if self._repetitiveness_score(s) > 0.3 else 0.0
        return penalty

    def _join_for_flow(self, sents: List[str]) -> List[str]:
        if len(sents) < 2:
            return sents
        joined = []
        i = 0
        while i < len(sents):
            cur = sents[i]
            if i + 1 < len(sents):
                nxt = sents[i + 1]
                if len(cur.split()) < 8 and len(nxt.split()) < 12:
                    cur = self._normalize_whitespace(cur.rstrip('.') + ", " + nxt[0].lower() + nxt[1:])
                    i += 2
                    joined.append(cur)
                    continue
            joined.append(cur)
            i += 1
        return joined

    def _adjust_readability(self, s: str) -> str:
        try:
            grade = textstat.flesch_kincaid_grade(s)
        except Exception:
            return s
        if grade > self.cfg.target_grade_level + 2:
            s = re.sub(r"[,;:]-?\s+", ", ", s)
            s = re.sub(r"\butilize\b", "use", s, flags=re.I)
            s = re.sub(r"\bendeavor\b", "try", s, flags=re.I)
        elif grade < self.cfg.target_grade_level - 3:
            s = re.sub(r"\.$", ", really.", s)
        return s

    def _tidy_sentence(self, s: str) -> str:
        s = self._normalize_whitespace(s)
        s = s[0].upper() + s[1:] if s and s[0].islower() else s
        s = re.sub(r"\s+([,.;!?])", r"\1", s)
        if not re.search(r"[.!?]\s*$", s):
            s += "."
        return s

    def _postprocess(self, sents: List[str]) -> str:
        return re.sub(r"\s+", " ", " ".join(sents)).strip()

    # -------------------- Metrics --------------------
    def _repetitiveness_score(self, text: str) -> float:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < 4:
            return 0.0
        bigrams = list(zip(tokens, tokens[1:]))
        total = len(bigrams)
        unique = len(set(bigrams))
        return 1.0 - unique / total if total else 0.0

    def _acceptable_change(self, original: str, new: str) -> bool:
        try:
            bleu = sacrebleu.sentence_bleu(new, [original]).score / 100.0
        except Exception:
            bleu = 0.6
        drift = 1.0 - bleu
        changed = drift > self.cfg.min_improvement_bleu
        not_too_far = drift < self.cfg.max_semantic_drift
        return changed and not_too_far

def _read_input(args: argparse.Namespace) -> str:
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            return f.read()
    data = sys.stdin.read()
    if data.strip():
        return data
    return args.text or ""


def _write_output(args: argparse.Namespace, text: str):
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        print(text)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text Humanizer Tool")
    p.add_argument("--text", type=str, default="", help="Direct input text")
    p.add_argument("--input-file", type=str, help="Path to input text file")
    p.add_argument("--out", type=str, help="Path to write humanized text")
    p.add_argument("--model", type=str, default=None, help="Paraphrase model name")
    p.add_argument("--aggressiveness", type=float, default=0.45, help="0..1: higher = more rephrasing")
    p.add_argument("--target-grade", type=int, default=9, help="Target Flesch-Kincaid grade")
    p.add_argument("--gpt2", type=str, default="gpt2", help="GPT-2 model id")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = HumanizerConfig(
        paraphrase_model=args.model,
        aggressiveness=args.aggressiveness,
        target_grade_level=args.target_grade,
        use_gpt2=args.gpt2,
    )
    th = TextHumanizer(cfg)
    text = _read_input(args)
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)
    out = th.humanize(text)
    _write_output(args, out)


if __name__ == "__main__":
    main()
