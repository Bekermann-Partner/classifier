import json
import os
import csv
import re
from transformers import pipeline
from pathlib import Path
from difflib import get_close_matches

def normalize_name_for_lookup(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    name = name.lower()

    return name

def get_best_fuzzy_match(target: str, dictionary_keys, threshold=0.8):
    matches = get_close_matches(target, dictionary_keys, n=1, cutoff=threshold)
    if matches:
        return matches[0]
    return None

def load_politicians_from_csv(politicians_csv_path):
    politicians_map = {}
    with open(politicians_csv_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            ext_id = row.get("ext_abgeordnetenwatch_id")
            first_name = row.get("first_name")
            last_name = row.get("last_name")

            if ext_id and first_name and last_name:
                full_name = normalize_name_for_lookup(f"{first_name} {last_name}")
                politicians_map[full_name] = ext_id

    return politicians_map


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512
)

def split_into_sentences(text):

    raw_sentences = text.split(".")
    return [s.strip() for s in raw_sentences if s.strip()]

def get_snippet(sentences, center_index, n_sentences_before=0, n_sentences_after=1):
    start = max(0, center_index - n_sentences_before)
    end = min(len(sentences), center_index + n_sentences_after + 1)
    snippet_sents = sentences[start:end]
    return ". ".join(snippet_sents).strip()

def relevant_snippets(text, politician, company, n_sentences_before=0, n_sentences_after=1):
    sentences = split_into_sentences(text)
    pol_lower = politician.lower()
    comp_lower = company.lower()

    snippet_list = []
    for i, sent in enumerate(sentences):
        s_lower = sent.lower()
        if pol_lower in s_lower and comp_lower in s_lower:
            snippet = get_snippet(sentences, i, n_sentences_before, n_sentences_after)
            snippet_list.append(snippet)
    return snippet_list

def classify_snippets(snippet_list):
    if not snippet_list:
        return None

    stars_values = []
    for snippet in snippet_list:
        result = sentiment_analyzer(snippet)[0]
        label = result["label"]  # e.g. "4 stars"
        stars = int(label.split()[0])
        stars_values.append(stars)

    if not stars_values:
        return None

    avg_stars = sum(stars_values) / len(stars_values)
    return round(avg_stars)

def classify_sentiment(text, politician, company, n_sentences_before=0, n_sentences_after=1):
    snippets = relevant_snippets(text, politician, company, n_sentences_before, n_sentences_after)
    return classify_snippets(snippets)


def process_json_file_to_csv(input_json_file, output_csv_file, politicians_csv_path,
                             n_sentences_before=0, n_sentences_after=1, fuzzy_threshold=0.8):

    politicians_map = load_politicians_from_csv(politicians_csv_path)
    all_politicians_keys = set(politicians_map.keys())

    fieldnames = ["politician_id", "url", "company", "stars"]

    with open(input_json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            content = entry["content"]

            raw_politicians = entry["politicians"]
            politicians = [p.strip() for p in raw_politicians.split(",") if p.strip()]

            raw_companies = entry["companies"]
            companies_list = [c.strip() for c in raw_companies.split(",") if c.strip()]

            for pol_name in politicians:
                normalized_json_name = normalize_name_for_lookup(pol_name)

                politician_ext_id = politicians_map.get(normalized_json_name)

                if politician_ext_id is None:
                    best_key = get_best_fuzzy_match(normalized_json_name, all_politicians_keys, threshold=fuzzy_threshold)
                    if best_key:
                        politician_ext_id = politicians_map[best_key]

                for company in companies_list:
                    sentiment_score = classify_sentiment(
                        content,
                        pol_name,
                        company,
                        n_sentences_before=n_sentences_before,
                        n_sentences_after=n_sentences_after
                    )

                    if sentiment_score is not None:
                        row = {
                            "politician_id": politician_ext_id,
                            "url": entry["url"],
                            "company": company,
                            "stars": sentiment_score
                        }
                        writer.writerow(row)


if __name__ == "__main__":
    print("Start")

    input_dir = Path("../data/input_jsons")
    output_dir = Path("../data/longer_classified_csv_files")
    politicians_csv_path = Path("../data/csv_files/Politician.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    n_sentences_before = 8
    n_sentences_after = 12
    fuzzy_threshold = 0.7

    json_files = input_dir.glob("*.json")

    for input_path in json_files:
        base_name = input_path.stem
        output_path = output_dir / f"{base_name}_classified.csv"

        print(f"Processing {input_path}...")

        try:
            process_json_file_to_csv(
                input_json_file=str(input_path),
                output_csv_file=str(output_path),
                politicians_csv_path=str(politicians_csv_path),
                n_sentences_before=n_sentences_before,
                n_sentences_after=n_sentences_after,
                fuzzy_threshold=fuzzy_threshold
            )
            print(f"Sentiment classifications saved to {output_path}.")
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

    print("All files have been processed.")
