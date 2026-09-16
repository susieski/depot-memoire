"""Lecture des étiquettes publiées et réglages communs aux figures du mémoire."""

import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

DEPOT = Path(__file__).resolve().parent.parent
DONNEES = DEPOT / "data"
SORTIES = DEPOT / "figures" / "sorties"

ANNOTATEURS = ["H1", "H2", "H3", "Grok", "Gemini"]
CLASSES = [0, 1, 2]
NOMS_CLASSES = ["classe 0\nnon sexiste", "classe 1\nhostile / direct", "classe 2\nimplicite / systémique"]

ENCRE = "#1a2a3a"
BLEU = "#2077b4"
GRIS = "#8c9aa5"
ROUGE = "#b5442e"
VERT = "#4a7c59"
SABLE = "#c9b04a"
COULEURS_CLASSES = ["#d9e1e8", BLEU, "#5a3d7a"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def entier(valeur):
    valeur = (valeur or "").strip()
    return None if valeur == "" else int(float(valeur))


def lire(nom):
    with open(DONNEES / nom, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def lire_lexical():
    return lire("corpus_lexical_etiquettes.csv")


def lire_evenementiel():
    return lire("corpus_evenementiel_etiquettes.csv")


def colonne(lignes, nom):
    return [entier(l[nom]) for l in lignes]


def kappa(a, b):
    paires = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(paires)
    if n == 0:
        return float("nan"), 0
    accord = sum(1 for x, y in paires if x == y) / n
    ca = collections.Counter(x for x, _ in paires)
    cb = collections.Counter(y for _, y in paires)
    hasard = sum(ca[c] * cb[c] for c in CLASSES) / n ** 2
    if hasard == 1:
        return float("nan"), n
    return (accord - hasard) / (1 - hasard), n


def accord_brut(a, b):
    paires = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return sum(1 for x, y in paires if x == y) / len(paires), len(paires)


def matrice_confusion(reference, prediction):
    m = np.zeros((3, 3), dtype=int)
    for r, p in zip(reference, prediction):
        if r is not None and p is not None:
            m[r][p] += 1
    return m


def f1_par_classe(matrice):
    scores = []
    for c in CLASSES:
        vrais = matrice[c][c]
        faux_positifs = matrice[:, c].sum() - vrais
        faux_negatifs = matrice[c, :].sum() - vrais
        precision = vrais / (vrais + faux_positifs) if vrais + faux_positifs else 0.0
        rappel = vrais / (vrais + faux_negatifs) if vrais + faux_negatifs else 0.0
        scores.append(2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0)
    return scores


def virgule(x, decimales=2):
    return f"{x:.{decimales}f}".replace(".", ",")


def enregistrer(figure, nom):
    SORTIES.mkdir(parents=True, exist_ok=True)
    chemin = SORTIES / nom
    figure.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(chemin.name)
