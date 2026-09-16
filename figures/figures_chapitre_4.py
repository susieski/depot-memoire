"""Figures 4.1 à 4.4. Demandent CORPUS_LEXICAL et CORPUS_EVENEMENTIEL."""

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from commun import DEPOT, ENCRE, SORTIES, enregistrer, virgule

TYPES = ["hostile", "anti_feministe_politique", "objectifiant_ou_sexuel", "incel", "paternaliste"]
NOMS = {
    "hostile": "Hostile",
    "anti_feministe_politique": "Antiféministe / politique",
    "objectifiant_ou_sexuel": "Objectifiant / sexuel",
    "incel": "Incel",
    "paternaliste": "Paternaliste",
}
PALETTE = {
    "hostile": "#c0392b",
    "anti_feministe_politique": "#7d3c98",
    "objectifiant_ou_sexuel": "#e67e22",
    "incel": "#2c3e50",
    "paternaliste": "#16816d",
}

ALPHA0 = 0.01
SEUIL_FREQUENCE = 15
PREMIERS_MOTS = 10
SEUIL_Z = 1.96

PIVOTS_LEXICAL = ["sexy", "folle"]
CONTEXTE = 38
LIGNES_KWIC = 12

COURONNEMENT = "00:45"
DEBUT_SOIREE = "23:30"
FIN_SOIREE = "00:55"
PAS_MINUTES = 5

VIDES = set("""le la les de des du un une et à a au aux en dans sur pour par avec sans sous chez vers entre
que qui quoi dont où est sont était été être etre suis es sommes êtes avoir ai as ont avait mais ou donc or ni car
ne pas plus très trop peu ce cet cette ces se sa son ses il elle ils elles je tu nous vous on y en
mon ma mes ton ta tes notre nos votre vos leur leurs lui moi toi soi d l c j n s m t qu si comme tout toute tous toutes
faire fait fais ça cela alors quand même aussi bien encore juste rien non oui va vais vas aller deux ans après avant
parce faut veut peut dit dire sais sait ah eh oh mdr ptdr lol jsp cest nest sest quil quelle
the and you for that this jai ya puis ainsi""".split())


def mots(texte):
    texte = re.sub(r"https?://\S+", " ", texte.lower())
    texte = re.sub(r"@\w+", " ", texte)
    texte = re.sub(r"#(\w+)", r"\1", texte)
    formes = re.findall(r"[a-zàâäéèêëîïôöùûüç][a-zàâäéèêëîïôöùûüç'\-]{2,}", texte)
    return [f for f in formes if f not in VIDES]


def nom_lisible(nom):
    try:
        nom = nom.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFC", nom).lower()


def termes_de_collecte(racine):
    formes = set()
    for fichier in racine.glob("*/*.json"):
        for forme in re.split(r"[^a-zàâäéèêëîïôöùûüç]+", nom_lisible(fichier.stem)):
            if len(forme) > 2:
                formes.add(forme)
    return formes


def chemin_lexical():
    brut = os.environ.get("CORPUS_LEXICAL", str(DEPOT.parent / "donnees_sources" / "CORPUS LEXICAL"))
    chemin = Path(brut)
    if not chemin.is_dir():
        sys.exit(f"Corpus lexical introuvable : {chemin}. Définissez CORPUS_LEXICAL.")
    return chemin


def chemin_evenementiel():
    brut = os.environ.get("CORPUS_EVENEMENTIEL",
                          str(DEPOT.parent / "donnees_sources" / "missfrance2026.json"))
    chemin = Path(brut)
    if not chemin.is_file():
        sys.exit(f"Corpus événementiel introuvable : {chemin}. Définissez CORPUS_EVENEMENTIEL.")
    return chemin


def charger_lexical():
    racine = chemin_lexical()
    textes = {t: [] for t in TYPES}
    vus = set()
    for fichier in sorted(racine.glob("*/*.json")):
        type_ = fichier.parent.name
        if type_ not in textes:
            continue
        with open(fichier, encoding="utf-8") as f:
            tweets = json.load(f)
        for tweet in tweets:
            identifiant = tweet.get("id")
            if identifiant in vus:
                continue
            vus.add(identifiant)
            textes[type_].append(tweet.get("text", "") or "")
    return textes


def table_lexicale(textes, exclues=frozenset()):
    compteurs = {t: Counter() for t in TYPES}
    for type_, liste in textes.items():
        for texte in liste:
            compteurs[type_].update(mots(texte))
    total = Counter()
    for compteur in compteurs.values():
        total.update(compteur)
    vocabulaire = sorted(forme for forme, n in total.items()
                         if n >= SEUIL_FREQUENCE and forme not in exclues)
    table = np.array([[compteurs[t][forme] for t in TYPES] for forme in vocabulaire], dtype=float)
    return vocabulaire, table, compteurs


def specificites(vocabulaire, compteurs):
    a0 = ALPHA0 * len(vocabulaire)
    resultats = {}
    for type_ in TYPES:
        dedans = compteurs[type_]
        dehors = Counter()
        for autre in TYPES:
            if autre != type_:
                dehors.update(compteurs[autre])
        n_dedans, n_dehors = sum(dedans.values()), sum(dehors.values())
        scores = []
        for forme in vocabulaire:
            x = dedans[forme] + ALPHA0
            y = dehors[forme] + ALPHA0
            ecart = np.log(x / (n_dedans + a0 - x)) - np.log(y / (n_dehors + a0 - y))
            z = ecart / np.sqrt(1 / x + 1 / y)
            scores.append((forme, z, dedans[forme], dehors[forme]))
        scores.sort(key=lambda s: -s[1])
        resultats[type_] = scores
    return resultats


def figure_specificites(resultats):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5.2))
    for ax, type_ in zip(axes, TYPES):
        premiers = resultats[type_][:PREMIERS_MOTS][::-1]
        ax.barh([m for m, *_ in premiers], [z for _, z, *_ in premiers], color=PALETTE[type_])
        ax.axvline(SEUIL_Z, color="#888888", ls="--", lw=1)
        ax.set_title(NOMS[type_], fontsize=13, color=PALETTE[type_], fontweight="bold")
        ax.set_xlabel("spécificité (z log-odds)", fontsize=10)
        ax.tick_params(labelsize=11)
    fig.suptitle("Spécificités lexicales des cinq parties du corpus lexical "
                 "(log-odds à prior de Dirichlet, Monroe et al., 2008)", fontsize=14, y=1.04)
    fig.tight_layout()
    enregistrer(fig, "figure_4_1_specificites.png")


def analyse_des_correspondances(table):
    proportions = table / table.sum()
    marges_lignes = proportions.sum(axis=1)
    marges_colonnes = proportions.sum(axis=0)
    attendu = np.outer(marges_lignes, marges_colonnes)
    residus = (proportions - attendu) / np.sqrt(attendu)
    u, valeurs, vt = np.linalg.svd(residus, full_matrices=False)
    inerties = valeurs ** 2
    parts = inerties / inerties.sum()
    lignes = u * valeurs / np.sqrt(marges_lignes)[:, None]
    colonnes = (vt.T * valeurs) / np.sqrt(marges_colonnes)[:, None]
    return lignes, colonnes, parts


def figure_correspondances(vocabulaire, table, resultats):
    lignes, colonnes, parts = analyse_des_correspondances(table)
    remarquables = set()
    for type_ in TYPES:
        remarquables.update(m for m, *_ in resultats[type_][:PREMIERS_MOTS])
    index = {forme: i for i, forme in enumerate(vocabulaire)}

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.axhline(0, color="#c8d2da", lw=1)
    ax.axvline(0, color="#c8d2da", lw=1)
    ax.scatter(lignes[:, 0], lignes[:, 1], s=8, color="#c8d2da", zorder=1)
    for forme in sorted(remarquables):
        i = index.get(forme)
        if i is None:
            continue
        ax.annotate(forme, (lignes[i, 0], lignes[i, 1]), fontsize=9.5, color=ENCRE,
                    ha="center", va="center", zorder=2)
    for j, type_ in enumerate(TYPES):
        ax.scatter(colonnes[j, 0], colonnes[j, 1], s=190, color=PALETTE[type_],
                   edgecolor="white", lw=2, zorder=3)
        ax.annotate(NOMS[type_], (colonnes[j, 0], colonnes[j, 1]), fontsize=12.5,
                    color=PALETTE[type_], fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 14), textcoords="offset points", zorder=4)
    ax.set_xlabel(f"axe 1 ({virgule(100 * parts[0], 1)} % de l'inertie)", fontsize=12)
    ax.set_ylabel(f"axe 2 ({virgule(100 * parts[1], 1)} % de l'inertie)", fontsize=12)
    ax.set_title("Analyse factorielle des correspondances mots × parties du corpus lexical",
                 fontsize=14, pad=14)
    fig.tight_layout()
    enregistrer(fig, "figure_4_2_correspondances.png")
    print(f"inertie du plan 1-2 : {virgule(100 * (parts[0] + parts[1]), 1)} %")


def concordances(textes, nom_fichier, titre):
    corpus = [t for liste in textes.values() for t in liste]
    lignes = []
    for pivot in PIVOTS_LEXICAL:
        motif = re.compile(rf"\b{pivot}\b", re.IGNORECASE)
        trouves = []
        for texte in corpus:
            propre = re.sub(r"@\w+", "", re.sub(r"https?://\S+", "[lien]", texte))
            propre = " ".join(propre.split())
            for occurrence in motif.finditer(propre):
                debut, fin = occurrence.span()
                gauche = propre[max(0, debut - CONTEXTE):debut]
                droite = propre[fin:fin + CONTEXTE]
                trouves.append((gauche, propre[debut:fin], droite))
                break
            if len(trouves) >= LIGNES_KWIC:
                break
        lignes.append((pivot, trouves))

    hauteur = 1.4 + sum(len(t) for _, t in lignes) * 0.26 + len(lignes) * 0.5
    fig, ax = plt.subplots(figsize=(13, hauteur))
    ax.axis("off")
    y = 0.93
    pas = 1.0 / (sum(len(t) for _, t in lignes) + 2 * len(lignes) + 1)
    for pivot, trouves in lignes:
        ax.text(0.5, y, f"« {pivot} »", fontsize=13, fontweight="bold", ha="center",
                color=ENCRE, family="DejaVu Sans")
        y -= 1.6 * pas
        for gauche, centre, droite in trouves:
            ax.text(0.485, y, gauche, fontsize=9, ha="right", family="DejaVu Sans Mono", color=ENCRE)
            ax.text(0.50, y, centre, fontsize=9, ha="center", family="DejaVu Sans Mono",
                    color="#b5442e", fontweight="bold")
            ax.text(0.515, y, droite, fontsize=9, ha="left", family="DejaVu Sans Mono", color=ENCRE)
            y -= pas
        y -= 0.8 * pas
    ax.set_title(titre, fontsize=13, pad=26)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def figure_activite():
    with open(chemin_evenementiel(), encoding="utf-8") as f:
        tweets = json.load(f)
    paris = timezone(timedelta(hours=1))
    horodatages = []
    for tweet in tweets:
        brut = tweet.get("createdAt")
        if not brut:
            continue
        instant = datetime.fromisoformat(brut.replace("Z", "+00:00")).astimezone(paris)
        horodatages.append(instant)
    if not horodatages:
        sys.exit("Aucun horodatage lisible dans le corpus événementiel.")

    reference = min(horodatages).date()
    def minutes(instant):
        depuis = instant - datetime.combine(reference, datetime.min.time(), tzinfo=paris)
        return depuis.total_seconds() / 60

    def borne(heure):
        h, m = (int(x) for x in heure.split(":"))
        return h * 60 + m + (24 * 60 if h < 12 else 0)

    debut, fin, sacre = borne(DEBUT_SOIREE), borne(FIN_SOIREE), borne(COURONNEMENT)
    tranches = np.arange(debut, fin + PAS_MINUTES, PAS_MINUTES)
    valeurs = [m for m in (minutes(h) for h in horodatages) if debut <= m <= fin]
    effectifs, _ = np.histogram(valeurs, bins=tranches)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    centres = tranches[:-1] + PAS_MINUTES / 2
    ax.bar(centres, effectifs, width=PAS_MINUTES * 0.86, color="#2077b4", edgecolor="white")
    ax.axvline(sacre, color="#b5442e", lw=2)
    ax.text(sacre + 2, max(effectifs) * 0.94, "couronnement", fontsize=12, color="#b5442e")
    graduations = np.arange(debut, fin + 1, 15)
    ax.set_xticks(graduations)
    ax.set_xticklabels([f"{int(g // 60) % 24:02d} h {int(g % 60):02d}" for g in graduations],
                       fontsize=11)
    ax.set_xlabel("heure de Paris", fontsize=12)
    ax.set_ylabel(f"tweets par tranche de {PAS_MINUTES} minutes", fontsize=12)
    ax.set_title("Activité du live-tweeting sous #MissFrance2026 (6 décembre 2025)",
                 fontsize=14, pad=12)
    fig.tight_layout()
    enregistrer(fig, "figure_4_4_activite_temporelle.png")
    print(f"{len(valeurs)} tweets dans la fenêtre {DEBUT_SOIREE} à {FIN_SOIREE}")


def main():
    racine = chemin_lexical()
    textes = charger_lexical()
    print(f"{sum(len(v) for v in textes.values())} tweets uniques répartis en {len(TYPES)} types")
    exclues = termes_de_collecte(racine)
    print(f"{len(exclues)} formes de requête écartées du calcul")
    vocabulaire, table, compteurs = table_lexicale(textes, exclues)
    print(f"{len(vocabulaire)} formes retenues (seuil de {SEUIL_FREQUENCE} occurrences)")

    resultats = specificites(vocabulaire, compteurs)
    SORTIES.mkdir(parents=True, exist_ok=True)
    with open(SORTIES / "specificites_par_type.csv", "w", encoding="utf-8-sig") as f:
        f.write("type,mot,z,occurrences_dans_le_type,occurrences_hors_type\n")
        for type_ in TYPES:
            for forme, z, dedans, dehors in resultats[type_]:
                f.write(f"{type_},{forme},{z:.4f},{dedans},{dehors}\n")

    figure_specificites(resultats)
    figure_correspondances(vocabulaire, table, resultats)
    concordances(textes, "figure_4_3_concordances.png",
                 f"Concordances (KWIC), corpus lexical, contexte de ± {CONTEXTE} caractères")
    figure_activite()


if __name__ == "__main__":
    main()
