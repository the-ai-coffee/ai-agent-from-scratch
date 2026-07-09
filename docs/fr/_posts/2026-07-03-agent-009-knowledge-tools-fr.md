---
layout: post
title: "Agent-009 : Donner du savoir à l'agent"
date: 2026-07-03
author: mikamboo
tags: [ia, agents, llm, claude, python, rag, embeddings, recherche]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools %}) | 🇫🇷 Français

Dans [agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling-fr %}), nous avons terminé le sous-système d'outils : un registre pour les tenir, une boucle pour les exécuter, un harnais qui leur survit quand ils tournent mal. Il est complet et solide — et chaque outil qu'on y a fait passer était factice. La calculatrice faisait des calculs, mais l'outil météo se contentait de lire trois lignes préenregistrées dans un dictionnaire. C'était voulu : pendant quatre étapes, la leçon était la *boucle*, alors les outils restaient des ébauches pour ne pas gêner.

On encaisse maintenant la mise. Cette étape accroche à la boucle les premiers outils avec de vraies tripes — et voici ce qu'il faut observer : **rien ne change dans l'agent.** La fonction `run` et le harnais `run_tool` sont ceux d'agent-008, au caractère près. Ce qui gagne en substance, c'est l'outil, pas l'agent. C'est tout l'intérêt d'avoir construit la boucle en premier : une capacité vraiment utile n'est qu'une nouvelle entrée dans le registre.

La capacité qu'on ajoute, c'est le *savoir* — précisément, un savoir sur lequel le modèle n'a jamais été entraîné. Notre corpus, ce sont cinq faits sur une entreprise qui n'existe pas :

```python
CORPUS = [
    "Nimbus Labs was founded in 2019 by Dara Okonkwo.",
    "The company's flagship product is Skyline, a weather-forecasting platform.",
    "Permanent staff receive 25 days of paid vacation each year.",
    "The office dog is a corgi named Biscuit.",
    "Nimbus Labs moved its headquarters from Lisbon to Porto in 2023.",
]
```

Comme Nimbus Labs est inventée, le modèle n'a aucun moyen de répondre à une question à son sujet, sauf en utilisant un outil pour la chercher. Ça en fait le test parfait pour vérifier que l'outil fonctionne vraiment : toute réponse correcte *doit* venir du corpus, pas de l'entraînement.

Et on va chercher de deux façons complètement différentes — parce que la vraie leçon de cette étape n'est pas « comment ajouter du savoir », mais *laquelle de deux méthodes rivales choisir, et quand.*

## Méthode 1 : un outil bête et une boucle intelligente

Le premier outil, `search`, est presque insultant de simplicité. Il découpe votre requête en mots et renvoie tout chunk qui partage au moins un mot exact :

```python
def search(query):
    terms = set(_tokenize(query))
    matches = [c for c in CORPUS if terms & set(_tokenize(c))]
    if not matches:
        return "No documents matched. Try different or broader keywords."
    return "\n".join(f"- {m}" for m in matches)
```

C'est tout. Aucun classement, aucune ruse — il compare des mots, rien de plus. Demandez-lui « holidays » et il ne trouvera *rien*, parce que le chunk dit « vacation », et « holidays » est une autre chaîne de caractères.

Un outil aussi grossier a l'air inutile. L'astuce, c'est qu'il ne travaille pas seul — il est assis à l'intérieur de la boucle construite dans agent-006, celle qui laisse le modèle appeler un outil, lire le résultat, et rappeler. Alors quand la première recherche revient vide, le modèle n'abandonne pas. Il lit « No documents matched », se dit *« essayons un autre mot »*, et recherche à nouveau avec « vacation » — qui, cette fois, tombe juste. L'intelligence n'est pas dans l'outil. **L'intelligence est dans l'itération**, et c'est la boucle qui rend l'itération possible.

C'est exactement comme ça que les agents de code trouvent des choses dans une base de code : ils n'ont pas d'index malin, ils ont `grep` et la patience de le lancer dix fois avec des motifs différents. Un outil de recherche bête dans une boucle intelligente bat un outil de recherche malin utilisé une seule fois. On appelle ça la **recherche agentique** : l'agent cherche, encore et encore, en affinant à mesure qu'il découvre ce qui existe.

## Méthode 2 : chercher le sens, pas les mots

La recherche agentique a une limite dure, et vous venez de la voir : elle ne trouve que les mots qui sont *là*. Si vous demandez « how much time off do staff get? » (combien de congés) et que le document dit « paid vacation », la recherche par mots-clés est impuissante — la requête et la réponse veulent dire la même chose mais ne partagent pas un seul mot. Aucune itération ne corrige un décalage de vocabulaire ; le modèle devrait *deviner* le mot exact qu'employait le document.

Le second outil, `retrieve`, résout ce décalage en cherchant sur le sens. L'idée repose sur un concept : l'**embedding** (plongement). Un embedding transforme un morceau de texte en une liste de nombres — un point dans l'espace — placé de sorte que les textes *qui parlent de la même chose* atterrissent près les uns des autres, même s'ils emploient des mots différents. « Time off » et « paid vacation » deviennent voisins non pas parce qu'ils partagent des lettres, mais parce qu'ils partagent un sens.

Les vrais embeddings viennent d'un modèle entraîné sur un océan de texte, avec des milliers de dimensions. Pour garder cette étape autonome et lisible, le nôtre est une miniature faite main : huit « concepts », et une petite table indiquant quels concepts chaque mot touche.

```python
CONCEPTS = ["founding", "people", "product", "weather", "timeoff", "location", "pet", "year"]

def embed(text):
    vector = np.zeros(len(CONCEPTS))
    for token in _tokenize(text):
        for concept in LEXICON.get(token, ()):
            vector[CONCEPTS.index(concept)] += 1.0
    return vector
```

« vacation » comme la paire « time »/« off » pointent vers le concept `timeoff`, donc les deux produisent des vecteurs penchant dans la même direction. C'est la seule pièce honnêtement truquée de toute la série : un vrai système *apprend* ces associations ; nous les avons écrites à la main pour que vous voyiez le mécanisme sans rien de caché. Tout ce qui est construit autour — le magasin, la similarité, la récupération — est exactement ce que fait un système de production.

Pour comparer deux vecteurs, on utilise la **similarité cosinus**, qui mesure l'angle entre eux — pointent-ils dans la même direction ? — et ignore leur longueur, si bien qu'une requête de trois mots et un chunk de dix mots peuvent quand même être une correspondance parfaite :

```python
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

Et le magasin lui-même est la plus petite chose qui mérite le nom de « base de données vectorielle » :

```python
class VectorStore:
    def __init__(self, chunks, embed):
        self.chunks = list(chunks)
        self.vectors = [embed(c) for c in self.chunks]   # on plonge une seule fois

    def search(self, query, top_k=2):
        q = self.embed(query)
        scored = [(cosine_similarity(q, v), c) for v, c in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(s, c) for s, c in scored[:top_k] if s > 0]
```

La seule vraie idée ici, c'est qu'on **embbed** chaque chunk *une fois*, au moment de l'import — l'étape d'« indexation » — pour que répondre à une requête revienne à « plonger la requête, comparer à tous ». Demandez « how much time off do staff get? » et, bien qu'elle ne partage aucun mot avec le chunk sur les congés, ce chunk revient en premier, parce que leurs vecteurs pointent dans la même direction.

Ce schéma récupérer-puis-répondre — chercher le texte pertinent, le tendre au modèle, le laisser répondre dessus — a un nom que vous avez presque sûrement entendu : le **RAG**, retrieval-augmented generation (génération augmentée par la récupération). Ôtez l'acronyme et c'est ce que vous venez de lire : un magasin de vecteurs, une comparaison cosinus, et les quelques meilleurs chunks recollés dans la conversation. Là non plus, aucune magie.

## Alors, lequel ?

Les deux outils répondent à « que dit la base de connaissances ? », et l'agent est libre de choisir l'un ou l'autre. La leçon de l'étape, c'est *quand chacun l'emporte* :

- **La recherche agentique** brille quand le corpus est *greppable* — les mots que vous voulez sont vraiment dans le texte — et que l'agent peut s'offrir quelques tours pour chasser. Elle n'a besoin d'aucun embedding, d'aucun index, d'aucune infrastructure en plus ; elle échange des appels d'API contre de la simplicité. C'est le schéma des agents de code.
- **Le RAG** brille quand la requête et les documents *veulent dire* la même chose avec *des mots différents*. Un seul coup, sans itérer, et il franchit le fossé de vocabulaire que la recherche par mots-clés ne peut pas franchir. Il vous coûte un modèle d'embedding et un index à entretenir.

Aucun n'est la « bonne » réponse. Un outil grossier dans une boucle intelligente, ou un outil intelligent appelé une seule fois — c'est un vrai choix d'ingénierie, et vous venez d'en construire les deux côtés.

## Le voir tourner

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-009-knowledge-tools/agent.py
```

Demandez « who founded Nimbus Labs? » et regardez apparaître une ligne `[tool] search(...)` ou `[tool] retrieve(...)`, puis une vraie réponse tirée du corpus. Puis demandez « what time off do employees get? » — formulé pour éviter le mot « vacation » — et voyez la récupération franchir le fossé qu'un mot-clé raterait. Enfin, demandez quelque chose que le corpus ne couvre pas, comme « what's the revenue? », et regardez l'agent faire chou blanc et le dire, au lieu d'inventer un chiffre.

Les tests verrouillent la distinction directement : `search("vacation")` trouve le fait par son mot exact, `retrieve("time off")` trouve le *même* fait par son sens, et un scénario scripté montre le modèle cherchant une première fois avec un mot absent, lisant le résultat vide, et cherchant à nouveau avec un meilleur — l'itération qui fait le travail.

## Et après

Notre agent a désormais de la mémoire, des outils, de la robustesse, et un vrai savoir où puiser. Mais cette mémoire d'agent-003 a un défaut qu'on a ignoré : elle grandit sans fin. À chaque tour, chaque résultat d'outil, chaque chunk récupéré s'empile sur la liste `messages` qu'on renvoie à chaque appel — et un jour, ça ne tiendra plus. Agent-010 affronte ce mur et ajoute la parade standard : quand l'historique devient trop long, l'agent tourne sa seule compétence — appeler le modèle — sur sa *propre* mémoire, en résumant les vieux tours pour faire de la place. L'arc de la mémoire ouvert dans agent-003 se referme là.
