"""Figures 6.1 à 6.4"""

import collections

import numpy as np
import matplotlib.pyplot as plt

from commun import (BLEU, ENCRE, GRIS, ROUGE, SABLE, VERT, colonne, enregistrer, kappa,
                    lire_evenementiel, lire_lexical, virgule)

NOMS_TYPES = {
    "hostile": "hostile",
    "anti_feministe_politique": "antiféministe\npolitique",
    "objectifiant_ou_sexuel": "objectifiant\nou sexuel",
    "incel": "incel",
    "paternaliste": "paternaliste",
}


def entonnoir(lignes, nom_fichier):
    reference = colonne(lignes, "reference_finale")
    total = len(lignes)
    sexistes = sum(1 for v in reference if v in (1, 2))
    faux = [l for l, v in zip(lignes, reference) if v == 0]
    relus = [(colonne([l], "Grok")[0], colonne([l], "Gemini")[0]) for l in faux]
    corriges = sum(1 for g, m in relus if g == 0 and m == 0)
    partiels = sum(1 for g, m in relus if (g == 0) != (m == 0))
    pieges = len(faux) - corriges - partiels

    fig, ax = plt.subplots(figsize=(12.5, 5))
    ax.barh(2, sexistes, 0.5, color=BLEU, edgecolor="white")
    ax.barh(2, len(faux), 0.5, left=sexistes, color=ROUGE, edgecolor="white")
    ax.text(sexistes / 2, 2, f"{sexistes} sexistes", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(sexistes + len(faux) / 2, 2,
            f"{len(faux)} faux positifs ({round(100 * len(faux) / total)} %)",
            ha="center", va="center", color="white", fontsize=12.5, fontweight="bold")
    ax.barh(1, corriges, 0.5, left=sexistes, color=VERT, edgecolor="white")
    ax.barh(1, partiels, 0.5, left=sexistes + corriges, color=SABLE, edgecolor="white")
    ax.barh(1, pieges, 0.5, left=sexistes + corriges + partiels, color=ROUGE, edgecolor="white")
    ax.text(sexistes + corriges / 2, 1, f"{corriges} corrigés par les deux modèles",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(total * 1.005, 1, f"{partiels} corrigés par un seul, {pieges} encore piégés",
            ha="left", va="center", color=ENCRE, fontsize=10.5)
    ax.set_yticks([2, 1])
    ax.set_yticklabels(["étape 1\nfiltrage lexical", "étape 2\nrelecture par modèle"], fontsize=12)
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(0.4, 2.6)
    ax.set_xlabel(f"nombre de tweets (échantillon lexical, n = {total})", fontsize=12)
    ax.set_title("L'entonnoir du filtrage : le mot capture large, le modèle relâche le bruit",
                 fontsize=14, pad=14)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def faux_positifs_par_type(lignes, nom_fichier):
    reference = colonne(lignes, "reference_finale")
    total = collections.Counter(l["type"] for l in lignes)
    faux = collections.Counter(l["type"] for l, v in zip(lignes, reference) if v == 0)
    types = sorted(total, key=lambda t: faux[t] / total[t], reverse=True)
    taux = [100 * faux[t] / total[t] for t in types]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(types))
    ax.bar(x, [total[t] for t in types], 0.62, color="#d9e1e8",
           label="tweets captés par le lexique")
    ax.bar(x, [faux[t] for t in types], 0.62, color=ROUGE,
           label="dont faux positifs (classe 0)")
    for i, t in enumerate(types):
        ax.text(i, total[t] + 1, f"{faux[t]}/{total[t]}", ha="center", va="bottom",
                fontsize=12, color=ENCRE)
        ax.text(i, faux[t] / 2, f"{round(taux[i])} %", ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NOMS_TYPES[t] for t in types], fontsize=11.5)
    ax.set_ylabel("nombre de tweets", fontsize=12)
    ax.set_ylim(0, max(total.values()) * 1.12)
    ax.set_title("Le bruit du lexique se concentre dans les types idéologiques", fontsize=14, pad=12)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def chute_de_accord(lignes, nom_fichier):
    moments = ["avant", "après"]
    couleurs = {"Gemini": BLEU, "Grok": GRIS}
    kappas = {}
    rappels = {}
    effectifs = {}
    for moment in moments:
        sous = [l for l in lignes if l["moment"] == moment]
        reference = colonne(sous, "reference")
        implicites = [i for i, v in enumerate(reference) if v == 2]
        effectifs[moment] = len(implicites)
        for modele in couleurs:
            prediction = colonne(sous, modele)
            kappas.setdefault(modele, []).append(kappa(reference, prediction)[0])
            retrouves = sum(1 for i in implicites if prediction[i] == 2)
            rappels.setdefault(modele, []).append(retrouves)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    gauche, droite = axes
    gauche.axhline(0, color=ENCRE, lw=1, ls=":")
    gauche.text(1.32, 0.012, "hasard", ha="right", va="bottom", fontsize=10, color=ENCRE)
    for modele, couleur in couleurs.items():
        valeurs = kappas[modele]
        gauche.plot([0, 1], valeurs, "-o", color=couleur, lw=2.5, ms=9, label=modele)
        gauche.text(-0.04, valeurs[0], virgule(valeurs[0], 3), ha="right", va="center",
                    fontsize=12, color=couleur)
        gauche.text(1.04, valeurs[1], virgule(valeurs[1], 3), ha="left", va="center",
                    fontsize=12, color=couleur)
    gauche.set_xticks([0, 1])
    gauche.set_xticklabels(["avant\nl'annonce", "après\nle couronnement"], fontsize=12)
    gauche.set_xlim(-0.35, 1.35)
    gauche.set_ylabel("kappa contre la majorité humaine", fontsize=12)
    gauche.set_title("L'accord machine s'effondre après le sacre", fontsize=13, pad=10)
    gauche.legend(fontsize=12, frameon=False, loc="upper right")

    x = np.arange(2)
    largeur = 0.36
    parts = {m: [rappels[m][i] / effectifs[moments[i]] if effectifs[moments[i]] else 0
                 for i in range(2)] for m in couleurs}
    droite.bar(x - largeur / 2, parts["Grok"], largeur, color=GRIS, hatch="//",
               edgecolor="white", label="Grok")
    droite.bar(x + largeur / 2, parts["Gemini"], largeur, color=BLEU, label="Gemini")
    for i, moment in enumerate(moments):
        droite.text(i - largeur / 2, parts["Grok"][i] + 0.012,
                    f"{rappels['Grok'][i]}/{effectifs[moment]}", ha="center", va="bottom",
                    fontsize=11, color=ENCRE)
        droite.text(i + largeur / 2, parts["Gemini"][i] + 0.012,
                    f"{rappels['Gemini'][i]}/{effectifs[moment]}", ha="center", va="bottom",
                    fontsize=11, color=ENCRE)
    droite.set_xticks(x)
    droite.set_xticklabels(["avant\nl'annonce", "après\nle couronnement"], fontsize=12)
    droite.set_ylim(0, 0.65)
    droite.set_ylabel("part du sexisme implicite retrouvée", fontsize=11.5)
    droite.set_title("Le sexisme bienveillant échappe surtout après le sacre", fontsize=13, pad=10)
    droite.legend(fontsize=12, frameon=False, loc="upper right")

    fig.suptitle("Corpus événementiel #MissFrance2026 : la soirée méliorative, angle mort des modèles",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def sens_des_erreurs(lignes, nom_fichier):
    reference = colonne(lignes, "reference_finale")
    modeles = ["Gemini", "Grok"]
    adoucit = []
    durcit = []
    for modele in modeles:
        prediction = colonne(lignes, modele)
        adoucit.append(sum(1 for r, p in zip(reference, prediction) if r == 1 and p in (0, 2)))
        durcit.append(sum(1 for r, p in zip(reference, prediction) if r == 0 and p in (1, 2)))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = np.arange(len(modeles))
    ax.barh(y, [-v for v in adoucit], 0.5, color=BLEU,
            label="adoucissement : hostile (1) rangé en 0 ou 2")
    ax.barh(y, durcit, 0.5, color=ROUGE,
            label="durcissement : anodin (0) rangé en 1 ou 2")
    for i in range(len(modeles)):
        ax.text(-adoucit[i] - 0.6, i, str(adoucit[i]), ha="right", va="center",
                fontsize=13, color=BLEU, fontweight="bold")
        ax.text(durcit[i] + 0.6, i, str(durcit[i]), ha="left", va="center",
                fontsize=13, color=ROUGE, fontweight="bold")
    ax.axvline(0, color=ENCRE, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(modeles, fontsize=13)
    borne = max(max(adoucit), max(durcit)) + 6
    ax.set_xlim(-borne, borne * 0.75)
    graduations = [g for g in range(-30, 31, 10) if -borne <= g <= borne * 0.75]
    ax.set_xticks(graduations)
    ax.set_xticklabels([abs(g) for g in graduations], fontsize=11)
    ax.set_xlabel("nombre de tweets mal classés (corpus lexical, n = 200)", fontsize=12)
    ax.set_title("Quand les modèles se trompent, ils adoucissent plus qu'ils ne durcissent",
                 fontsize=13.5, pad=12)
    ax.legend(fontsize=11.5, frameon=False, loc="lower right")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def rappel_lexique(lexical, evenementiel, nom_fichier):
    """Bruit du lexique en haut, rappel en bas. terme_lexique vient de marquer_lexique.py."""
    reference = colonne(lexical, "reference_finale")
    bruit = sum(1 for v in reference if v == 0)
    sexistes_lex = len(reference) - bruit

    sexistes = [l for l in evenementiel if l["reference"] in ("1", "2")]
    implicites = [l for l in sexistes if l["reference"] == "2"]
    captes = sum(1 for l in sexistes if l["terme_lexique"] == "1")
    manques = len(sexistes) - captes

    barres = [
        (1, "précision\nles 200 tweets que\nle lexique ramène",
         sexistes_lex, bruit,
         f"{sexistes_lex} relèvent du sexisme",
         f"{bruit} faux positifs   {round(100 * bruit / len(reference))} %"),
        (0, "rappel\nles 35 tweets sexistes\nde #MissFrance2026",
         captes, manques,
         f"{captes} capté",
         f"{manques} sexismes manqués   {round(100 * manques / len(sexistes))} %"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    for y, _, bon, mauvais, texte_bon, texte_mauvais in barres:
        total = bon + mauvais
        part = 100 * bon / total
        ax.barh(y, part, 0.3, color=BLEU, edgecolor="white", linewidth=1.6)
        ax.barh(y, 100 - part, 0.3, left=part, color=ROUGE, edgecolor="white",
                linewidth=1.6, hatch="//")
        cadre = dict(facecolor=ROUGE, edgecolor="none", pad=3.5)
        if part > 12:
            ax.text(part / 2, y, texte_bon, ha="center", va="center", color="white",
                    fontsize=12, fontweight="bold")
        else:
            ax.text(part + 1.2, y + 0.27, texte_bon, ha="left", va="center",
                    color=BLEU, fontsize=11.5, fontweight="bold")
        ax.text(part + (100 - part) / 2, y, texte_mauvais, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold", bbox=cadre)

    ax.text(100, -0.38, "les 32 tweets de sexisme implicite sont manqués sans exception",
            ha="right", va="center", fontsize=10.5, color=ROUGE, style="italic")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([b[1] for b in barres], fontsize=11)
    ax.set_ylim(-0.58, 1.34)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{g} %" for g in range(0, 101, 20)], fontsize=10.5)
    ax.set_xlabel("part de chaque ensemble", fontsize=11.5)
    ax.set_title("Hors du corpus qu'il a servi à construire, le lexique retrouve "
                 "un tweet sexiste sur trente-cinq", fontsize=13, pad=14)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def main():
    lexical = lire_lexical()
    evenementiel = lire_evenementiel()
    entonnoir(lexical, "figure_6_1_entonnoir.png")
    faux_positifs_par_type(lexical, "figure_6_2_fp_par_type.png")
    chute_de_accord(evenementiel, "figure_6_3_chute_accord.png")
    sens_des_erreurs(lexical, "figure_6_4_sens_des_erreurs.png")
    rappel_lexique(lexical, evenementiel, "figure_6_5_rappel_lexique.png")


if __name__ == "__main__":
    main()
