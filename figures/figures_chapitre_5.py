"""Figures 5.1 à 5.7"""

import itertools

import numpy as np
import matplotlib.pyplot as plt

from commun import (ANNOTATEURS, CLASSES, COULEURS_CLASSES, ENCRE, BLEU, GRIS, NOMS_CLASSES,
                    accord_brut, colonne, enregistrer, f1_par_classe, kappa,
                    lire_evenementiel, lire_lexical, matrice_confusion, virgule)


def distribution(lignes, nom_fichier, titre):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ANNOTATEURS))
    bas = np.zeros(len(ANNOTATEURS))
    for classe in CLASSES:
        hauteurs = [sum(1 for v in colonne(lignes, a) if v == classe) for a in ANNOTATEURS]
        ax.bar(x, hauteurs, 0.62, bottom=bas, color=COULEURS_CLASSES[classe],
               edgecolor="white", label=f"classe {classe}")
        for i, h in enumerate(hauteurs):
            if h:
                ax.text(i, bas[i] + h / 2, str(h), ha="center", va="center", fontsize=12,
                        color=ENCRE if classe == 0 else "white", fontweight="bold")
        bas += np.array(hauteurs)
    manquants = [sum(1 for v in colonne(lignes, a) if v is None) for a in ANNOTATEURS]
    if any(manquants):
        ax.bar(x, manquants, 0.62, bottom=bas, color="white", edgecolor=ENCRE,
               hatch="///", label="sans réponse")
        for i, m in enumerate(manquants):
            if m:
                ax.text(i, bas[i] + m + 2, f"{m} sans réponse", ha="center", va="bottom",
                        fontsize=10, color=ENCRE)
    ax.set_ylim(0, len(lignes) * 1.09)
    ax.set_xticks(x)
    ax.set_xticklabels(ANNOTATEURS, fontsize=12.5)
    ax.set_ylabel("nombre de tweets", fontsize=12)
    ax.set_title(titre, fontsize=14, pad=12)
    ax.legend(fontsize=11.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def kappas_par_paire(lignes, nom_fichier, titre):
    paires = list(itertools.combinations(ANNOTATEURS, 2))
    valeurs = [kappa(colonne(lignes, a), colonne(lignes, b))[0] for a, b in paires]
    ordre = np.argsort(valeurs)
    paires = [paires[i] for i in ordre]
    valeurs = [valeurs[i] for i in ordre]
    humain = [a.startswith("H") and b.startswith("H") for a, b in paires]
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(paires))
    ax.barh(y, valeurs, 0.62,
            color=[ENCRE if h else BLEU for h in humain],
            edgecolor="white")
    for i, v in enumerate(valeurs):
        ax.text(v + 0.008, i, virgule(v, 3), va="center", fontsize=11.5, color=ENCRE)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} / {b}" for a, b in paires], fontsize=12)
    ax.set_xlabel("kappa de Cohen", fontsize=12)
    ax.set_xlim(0, max(valeurs) * 1.18)
    ax.set_title(titre, fontsize=14, pad=12)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def confusions(lignes, reference, nom_fichier, titre, legende_reference):
    ref = colonne(lignes, reference)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, modele in zip(axes, ["Grok", "Gemini"]):
        pred = colonne(lignes, modele)
        m = matrice_confusion(ref, pred)
        taux, n = accord_brut(ref, pred)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
        ax.set_xticks(CLASSES)
        ax.set_yticks(CLASSES)
        ax.set_xticklabels(CLASSES, fontsize=12)
        ax.set_yticklabels(CLASSES, fontsize=12)
        ax.set_xlabel(f"classe prédite par {modele}", fontsize=13)
        ax.set_ylabel(legende_reference, fontsize=13)
        ax.set_title(f"{modele} (n = {n}, accord = {virgule(taux, 3)})", fontsize=14, pad=12)
        for i in CLASSES:
            for j in CLASSES:
                ax.text(j, i, str(m[i][j]), ha="center", va="center", fontsize=15,
                        fontweight="bold", color="white" if m[i][j] > m.max() * 0.55 else ENCRE)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(titre, fontsize=16, y=1.02)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def f1(lignes, reference, nom_fichier, titre, legende_axe):
    ref = colonne(lignes, reference)
    scores = {m: f1_par_classe(matrice_confusion(ref, colonne(lignes, m))) for m in ["Grok", "Gemini"]}
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(3)
    largeur = 0.38
    barres = [
        ax.bar(x - largeur / 2, scores["Grok"], largeur, label="Grok", color=GRIS,
               hatch="//", edgecolor="white", lw=0),
        ax.bar(x + largeur / 2, scores["Gemini"], largeur, label="Gemini", color=BLEU),
    ]
    for serie, modele in zip(barres, ["Grok", "Gemini"]):
        for rect, valeur in zip(serie, scores[modele]):
            ax.text(rect.get_x() + rect.get_width() / 2, valeur + 0.012, virgule(valeur),
                    ha="center", va="bottom", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(legende_axe, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(NOMS_CLASSES, fontsize=12.5)
    ax.set_title(titre, fontsize=15, pad=12)
    ax.legend(fontsize=13, frameon=False, loc="upper right")
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def main():
    lexical = lire_lexical()
    evenementiel = lire_evenementiel()

    distribution(lexical, "figure_5_1_distribution_classes.png",
                 "Distribution des classes attribuées, corpus lexical (n = 200)")
    kappas_par_paire(lexical, "figure_5_2_kappa_par_paire.png",
                     "Kappa de Cohen pour chaque paire d'annotateurs, corpus lexical")
    confusions(lexical, "reference_finale", "figure_5_3_confusions_llm.png",
               "Matrices de confusion des modèles contre la référence finale", "référence finale")
    f1(lexical, "majorite_humaine", "figure_5_4_f1_par_classe.png",
       "F1 par classe : l'implicite reste le point dur des deux modèles",
       "F1 contre la majorité humaine")
    distribution(evenementiel, "figure_5_5_distribution_evenementiel.png",
                 "Distribution des classes attribuées, corpus événementiel (n = 200)")
    confusions(evenementiel, "reference", "figure_5_6_confusions_evenementiel.png",
               "Matrices de confusion des modèles, corpus événementiel", "majorité humaine")
    f1(evenementiel, "reference", "figure_5_7_f1_evenementiel.png",
       "F1 par classe, corpus événementiel", "F1 contre la majorité humaine")


if __name__ == "__main__":
    main()
