"""Step 1 - clean MAESTRO v3.0.0 metadata for the una corda study.

Usage:
    python -I 01_clean_metadata.py <maestro-v3.0.0_performer.csv> <out_dir>

Output:
    <out_dir>/metadata_clean.csv  one row per performance (1,276 rows)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

perf_csv, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(perf_csv)

# 1. Pianist names: truncated / variant spellings found in the performer table.
#    's' (2011, Canada) sits in the same recording session as Tristan Teo; assigning it to him
#    restores the R2-subset-of-R1 nesting for 2011, so we treat it as Tristan Teo.
NAME_FIX = {
    "ean-Selim Abdelmoula": "Jean-Selim Abdelmoula",
    "u Yeon Kim": "Su Yeon Kim",
    "Frank Duepree": "Frank Dupree",
    "s": "Tristan Teo",
}
df["performer_raw"] = df["performer"]
df["pianist"] = df["performer"].str.strip().replace(NAME_FIX)

# 2. Nationality: strip whitespace, merge country names and adjectives.
NAT_FIX = {
    "American": "USA", "United States": "USA", "Russian": "Russia", "Chinese": "China",
    "P. R. China": "China", "Canadian": "Canada", "South Korean": "South Korea",
    "S. Korea": "South Korea", "Taiwanese": "Taiwan", "German": "Germany", "Polish": "Poland",
    "Belarussian": "Belarus", "Ukranian": "Ukraine", "Italian": "Italy", "Israeli": "Israel",
    "Georgian": "Georgia", "Kazakstan": "Kazakhstan", "Lithuanian": "Lithuania",
    "Romanian": "Romania", "Dutch": "Netherlands", "New Zealander": "New Zealand",
}
df["nationality_raw"] = df["nationality"]
df["nationality"] = df["nationality"].str.strip().replace(NAT_FIX)

# 3. Composer errors inherited from MAESTRO.
df["composer"] = df["canonical_composer"]
df.loc[df.canonical_title.eq("Barcarolle, Op. 60"), "composer"] = "Frédéric Chopin"
df.loc[df.canonical_title.str.contains('Op. 35, "Eroica"', regex=False), "composer"] = "Ludwig van Beethoven"
df["is_transcription"] = df["composer"].str.contains(" / ")

# 4. Style period, by the arranger for transcriptions.
BAROQUE = {"Johann Sebastian Bach", "Domenico Scarlatti", "George Frideric Handel", "Henry Purcell",
           "Jean-Philippe Rameau", "Johann Pachelbel", "Orlando Gibbons", "Antonio Soler"}
CLASSICAL = {"Joseph Haydn", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Muzio Clementi"}
MODERN = {"Claude Debussy", "Alban Berg", "Alexander Scriabin", "Sergei Rachmaninoff", "Leoš Janáček",
          "George Enescu", "Nikolai Medtner", "Percy Grainger", "Isaac Albéniz", "Leopold Godowsky",
          "Ferruccio Busoni", "Egon Petri", "Myra Hess", "Vladimir Horowitz", "György Cziffra",
          "Vyacheslav Gryaznov", "Mikhail Pletnev", "Moritz Moszkowski", "Alfred Grünfeld",
          "Camille Saint-Saëns"}
def era(c):
    c = c.split(" / ")[-1]
    return ("Baroque" if c in BAROQUE else "Classical" if c in CLASSICAL
            else "Modern" if c in MODERN else "Romantic")
df["era"] = df["composer"].map(era)

# 5. Competition structure: junior editions per the competition's published history.
JUNIOR = {2008, 2011, 2013, 2015, 2017}
df["group"] = np.where(df.year.isin(JUNIOR), "junior", "senior")
def round_of(f):
    if "Chamber" in f:
        return "Chamber"
    m = re.search(r"_(R\d)_", f)
    return m.group(1) if m else None
df["round"] = df.midi_filename.map(round_of)

# advanced = pianist appears in R2 of the same edition (only defined for pianists with R1 files)
r2 = set(map(tuple, df[df["round"] == "R2"][["year", "pianist"]].values))
r1 = set(map(tuple, df[df["round"] == "R1"][["year", "pianist"]].values))
df["advanced_R2"] = [((y, p) in r2) if (y, p) in r1 else np.nan for y, p in zip(df.year, df.pianist)]

# Top-3 prize winners per edition (Wikipedia, competition history table), matched by cleaned name.
LAUREATES = {
    2004: ["Jie Chen", "Yung Wook Yoo", "Denis Evstioukhine"],
    2006: ["Edisher Savitski", "Ryo Yanagitani", "Victoria Korchinskaya-Kogan"],
    2009: ["Alessandro Taverna", "Pavel Yeletskiy", "Helene Tysman"],
    2014: ["Peter Friis Johannson", "Alexey Chernov", "Frank Dupree"],
    2018: ["KaJeng Wong", "Timur Mustakimov", "Su Yeon Kim"],
    2008: ["Frank Dupree", "Nansong Huang", "Jan Lisiecki", "Osip Nikiforov"],
    2011: ["Aristo Sham", "Tristan Teo", "Su Yeon Kim"],
    2013: ["Eric Lu", "Carmen Knoll", "Yuanfan Yang"],
    2015: ["Nathan Lee", "Elliot Wuu", "Arsenii Mun"],
    2017: ["Shuan Hern Lee", "Youl Sun", "Andrew Li"],
}
df["laureate_top3"] = [p in LAUREATES.get(y, []) for y, p in zip(df.year, df.pianist)]

# 6. Repeat entrants and transition type.
yrs = df.groupby("pianist").year.apply(lambda s: sorted(set(s)))
def transition(p):
    ys = yrs[p]
    if len(ys) < 2:
        return "single"
    g = ["junior" if y in JUNIOR else "senior" for y in ys]
    return f"{g[0]}->{g[-1]}"
df["repeat_type"] = df.pianist.map(transition)

ids = {p: i for i, p in enumerate(sorted(df.pianist.unique()))}
df["pianist_id"] = df.pianist.map(ids)
cols = ["midi_filename", "year", "group", "round", "advanced_R2", "laureate_top3", "pianist", "pianist_id",
        "performer_raw", "repeat_type", "nationality", "nationality_raw", "composer", "canonical_composer",
        "canonical_title", "canonical_title_label", "era", "is_transcription", "duration", "split"]
df[cols].to_csv(out_dir / "metadata_clean.csv", index=False)

# Checks printed for the log
print("rows", len(df), "| pianists", df.pianist.nunique(), "| nationalities", df.nationality.nunique())
for y, g in df.groupby("year"):
    s = {r: set(g[g["round"] == r].pianist) for r in ["R1", "R2", "R3"]}
    if s["R1"]:
        print(y, "R2 subset of R1:", s["R2"] <= s["R1"], "| R3 not in R2:", sorted(s["R3"] - s["R2"]))
print("repeat types:", df.drop_duplicates("pianist").repeat_type.value_counts().to_dict())
print("laureates matched:", df[df.laureate_top3].drop_duplicates(["year", "pianist"]).shape[0], "of",
      sum(len(v) for v in LAUREATES.values()))
