---
layout: post
title: "Agent-008 : La solidité avant la substance"
date: 2026-07-02
author: mikamboo
tags: [ai, agents, llm, claude, python, tools, errors, robustness]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling %}) | 🇫🇷 Français

Dans [agent-007]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch-fr %}), nous avons donné à l'agent une boîte à outils : un registre d'outils, un dispatch par nom, et un modèle libre de choisir l'outil qui convient — ou aucun. Cela fonctionne à merveille, tant que tout le monde se tient bien. Et c'est l'hypothèse que nous faisions en silence depuis l'étape 005 : que chaque appel d'outil arrive bien formé, et que chaque outil s'exécute jusqu'au bout.

Les vrais appels rompent cette promesse de trois façons différentes. Le modèle peut nommer un outil que nous n'avons jamais enregistré. Il peut nommer un vrai outil mais envoyer des arguments qui ne conviennent pas. Ou l'appel peut être parfait et *l'outil lui-même* peut exploser en plein vol. Dans agent-007, deux de ces trois cas font tomber tout l'agent en plein tour — une seule mauvaise requête du modèle et le programme plante, tout simplement.

Cette étape corrige cela. Pas de nouvel outil, pas de nouveau pouvoir — juste de la solidité. C'est l'étape la moins spectaculaire de la série, et celle qui sépare une démo de quelque chose qu'on oserait vraiment laisser tourner.

## Trois façons de rater une commande

Imaginez un serveur qui porte les commandes des clients vers la cuisine. Trois choses peuvent mal tourner avec une commande :

1. Le client commande un plat qui n'est pas à la carte. *(Outil inconnu : le modèle demande `"teleporter"` et notre registre n'a pas d'entrée de ce nom.)*
2. Le client commande un vrai plat mais embrouille les détails — « le poisson, à point, sans la... euh » — et la cuisine n'y comprend rien. *(Mauvais arguments : le modèle appelle `get_weather` avec `{"town": "Paris"}` alors que la fonction attend `city`.)*
3. La commande est parfaitement claire, mais la poêle prend feu. *(L'outil lève une exception : `calculator` reçoit `"1 / 0"` et la division par zéro explose en plein calcul.)*

Voici la question clé : dans chacun de ces cas, que doit faire le *serveur* ? Certainement pas s'effondrer par terre — ce que faisait notre agent jusqu'à maintenant. Le serveur retourne à la table et dit ce qui s'est passé : « nous n'avons pas ça », « la cuisine n'a pas compris », « il y a eu un incident avec votre plat ». Puis c'est le *client* qui décide : reformuler la commande, choisir autre chose, ou renoncer et s'excuser auprès de ses invités.

C'est tout le design de cette étape. Les erreurs ne tuent pas la boucle ; elles la retraversent en sens inverse, comme de l'information, vers le seul participant qui peut réellement décider de la suite : le modèle.

## L'erreur devient un résultat

[`agents/agent-008-malformed-tool-call-handling/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-008-malformed-tool-call-handling/agent.py)

Le changement est concentré dans `run_tool`, qui renvoie désormais une paire — le texte du résultat, plus un drapeau disant s'il s'agit d'une erreur :

```python
def run_tool(name, tool_input):
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}", True
    if not isinstance(tool_input, dict):
        return f"Error: arguments for {name!r} must be an object, ...", True
    try:
        return entry["function"](**tool_input), False
    except TypeError as error:
        return f"Error: bad arguments for {name!r}: {error}", True
    except Exception as error:
        return f"Error: tool {name!r} raised {type(error).__name__}: {error}", True
```

Lisez-le de haut en bas et vous reconnaîtrez les trois pannes du restaurant. Pas à la carte : la recherche dans le registre échoue. Détails embrouillés : les arguments ne correspondent pas à la signature de la fonction, ce que Python signale par un `TypeError`. Feu en cuisine : l'outil a levé quelque chose, n'importe quoi, pendant son exécution — le `except Exception` final l'attrape quoi que ce soit. Dans tous les cas, la fonction *retourne* — elle ne laisse jamais une exception s'échapper dans la boucle.

La boucle fait alors une petite chose nouvelle avec ce drapeau. Un appel raté produit quand même un `tool_result`, exactement comme un appel réussi, mais marqué :

```python
tool_result = {
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": result,
}
if is_error:
    tool_result["is_error"] = True
```

Ce champ `is_error: True` fait partie du vocabulaire de l'API, ce n'est pas une invention à nous : il dit au modèle « c'est ta requête qui a mal tourné — pas le monde, ta requête ». Et parce que l'erreur revient comme un résultat ordinaire, tout ce que nous avons construit en 006 et 007 s'y applique sans changement : elle est ajoutée à la conversation, le modèle la lit à l'appel suivant, et la boucle continue de tourner. Le modèle voit `Error: bad arguments for 'get_weather': ... unexpected keyword argument 'town'` et fait ce qu'on espère : il rappelle avec `city`. Nous n'avons écrit aucune logique de réessai. La boucle *est* la logique de réessai — renvoyer les résultats et rappeler le modèle, c'est ce qu'elle fait depuis l'étape 006. Nous avons simplement cessé d'en exempter les échecs.

## L'outil est devenu plus bête, exprès

Un changement semble aller dans le mauvais sens : la calculatrice d'agent-007 attrapait ses propres erreurs et renvoyait de polies chaînes `"Error: ..."`. Celle de cette étape ne le fait plus — donnez-lui une expression invalide et elle lève une exception, sans filet.

C'est délibéré, et c'est la deuxième leçon de l'étape. Si la sécurité vit à l'intérieur de chaque outil, alors la boucle n'est solide que dans la mesure de l'outil le plus négligemment écrit du registre — et dès que les outils viennent d'ailleurs (un serveur MCP publié par quelqu'un d'autre, par exemple), impossible de tous les auditer. La sécurité déménage donc hors des outils, vers le harnais : `run_tool` enveloppe *chaque* outil dans la même protection, et un outil a désormais le droit d'être négligé sans mettre l'agent en danger. La boucle ne fait pas confiance à ses outils, et c'est précisément pour cela qu'on peut y brancher n'importe quoi.

## Le voir fonctionner

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-008-malformed-tool-call-handling/agent.py
```

Demandez « combien font 1 / 0 ? ». Vous verrez quelque chose comme :

```
[tool-error] calculator({'expression': '1 / 0'}) -> Error: tool 'calculator' raised ZeroDivisionError: division by zero
Agent> La division par zéro n'est pas définie, cette question n'a donc pas de réponse...
```

L'outil a réellement explosé — et au lieu d'une pile d'erreurs Python, vous avez eu une phrase calme. La ligne `[tool-error]`, c'est le serveur qui revient à la table.

Les tests verrouillent chaque mode de panne avec le faux client scénarisé des étapes précédentes : un nom d'outil inconnu, une mauvaise clé d'argument, et un outil trafiqué pour lever une exception — chacun vérifiant que la boucle survit jusqu'à l'appel suivant du modèle et que le message qu'il reçoit porte `is_error: True`. Un dernier test scénarise l'arc complet du rétablissement : mauvais arguments, erreur renvoyée, appel corrigé, vraie réponse.

## Et après

Le sous-système d'outils est maintenant complet : un registre pour ranger les outils, une boucle pour les exécuter, et un harnais qui leur survit. Ce qui veut dire que nous pouvons enfin nous offrir ce que nous repoussons depuis l'étape 005 — un outil avec un vrai contenu. Tous les outils jusqu'ici étaient des maquettes, parce que la leçon était la boucle, pas l'outil. Agent-009 construit les premiers outils qui *font* vraiment quelque chose : donner à l'agent des connaissances sur lesquelles il n'a pas été entraîné, par deux voies rivales — la recherche itérative par mots-clés, où c'est la boucle elle-même qui fait l'intelligence, contre la récupération sémantique en un coup, le motif que l'industrie appelle RAG. La boucle est construite et endurcie ; il est temps d'y accrocher quelque chose de réel.
