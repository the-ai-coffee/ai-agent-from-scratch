---
layout: post
title: "Agent-007 : Une boîte à outils, pas un outil"
date: 2026-07-01
author: mikamboo
tags: [ia, agents, llm, claude, python, outils, dispatch, mcp]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch %}) | 🇫🇷 Français

Dans agent-006, nous avons fermé la boucle : le modèle réclame un outil, nous l'exécutons, nous renvoyons le résultat, et le modèle répond. Cette boucle *est* l'agent — nous l'avions dit, et nous le pensions. Mais notre agent ne possède qu'un seul outil, et pire, le nom de cet outil est câblé en dur dans le code : quelque part dans la boucle se trouve une ligne qui dit, en substance, « si le modèle a demandé la calculatrice, exécute la calculatrice ». Cela fonctionne très bien pour un outil. Cela cesse d'être une architecture et commence à devenir un fouillis dès qu'on en ajoute un deuxième.

Cette étape ajoute le deuxième outil — et avec lui, la chose qu'un seul outil ne vous force jamais à construire : un *registre*.

## Pourquoi deux, c'est différent d'un

Pensez à un tiroir de cuisine. Si vous possédez un seul couteau, vous n'avez pas besoin de système ; le couteau vit dans le tiroir et votre main sait où aller. Possédez douze ustensiles et soudain il vous faut le range-couverts : des emplacements étiquetés, un par ustensile, pour que trouver le fouet ne signifie pas tout retourner.

Un seul outil nous permettait de tricher de la même façon. L'agent n'avait besoin de *rien chercher* — il n'y avait qu'une seule réponse possible à « quelle fonction dois-je exécuter ? », alors nous l'avions écrite directement dans la boucle. Avec deux outils, « lequel ? » devient une vraie question, et elle se pose à deux endroits différents :

1. **Le modèle** doit savoir ce qui est disponible, pour choisir le bon outil — ou décider que la question n'a besoin d'aucun outil.
2. **Notre code** doit savoir, quand une requête revient en nommant un outil, à quelle fonction Python ce nom correspond.

Les deux questions ont la même réponse, elle devrait donc vivre à un seul endroit. Cet endroit, c'est le registre.

## Le registre

[`agents/agent-007-multi-tool-dispatch/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-007-multi-tool-dispatch/agent.py)

```python
TOOL_REGISTRY = {
    "calculator": {
        "function": calculator,
        "schema": {
            "description": "Evaluate a basic arithmetic expression...",
            "input_schema": {...},
        },
    },
    "get_weather": {
        "function": get_weather,
        "schema": {
            "description": "Get the current weather for a city.",
            "input_schema": {...},
        },
    },
}
```

C'est un simple dictionnaire. Chaque entrée associe le *nom* d'un outil aux deux choses que quiconque pourrait vouloir savoir à son sujet : la **fonction** que nous exécutons quand il est appelé, et le **schéma** — la description lisible et la spécification des arguments — qui dit au modèle à quoi sert l'outil et comment le demander.

Tout le reste du fichier est désormais *dérivé* de ce dictionnaire. La liste des outils que nous annonçons au modèle, ce sont simplement les schémas du registre, chacun étiqueté de son nom :

```python
TOOLS = [{"name": name, **entry["schema"]} for name, entry in TOOL_REGISTRY.items()]
```

Et le dispatch — le code qui répond « quelle fonction ? » quand une requête d'outil revient — est une consultation de dictionnaire :

```python
def run_tool(name, tool_input):
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Error: unknown tool {name!r}"
    return entry["function"](**tool_input)
```

Comparez avec la version d'agent-006, qui disait `if name == "calculator": return calculator(...)`. La chaîne de `if` a disparu, remplacée par une consultation qui fonctionne pour deux outils, ou dix, sans jamais grossir. Ajouter un outil à cet agent signifie désormais ajouter une entrée à un dictionnaire. La boucle — le code que nous avions appelé le battement de cœur la dernière fois — ne change pas du tout. C'était la promesse de construire la boucle d'abord : tout ce qui vient après est un outil qu'on y accroche.

## Le nouvel outil est volontairement banal

Le deuxième outil, `get_weather`, cherche une ville dans une petite table de bulletins météo préenregistrés et renvoie une phrase. Trois villes, des températures codées en dur, aucune API. C'est voulu. Si nous l'avions branché sur un vrai service météo, la partie intéressante de cette étape — le choix, le dispatch — serait enfouie sous des appels HTTP et des clés d'API. Les outils sont des coquilles vides parce que la leçon, c'est l'étagère, pas ce qui est posé dessus. (C'est à l'étape 009 qu'un outil aura enfin un vrai contenu.)

## Choisir — y compris ne rien choisir

Avec deux outils annoncés, le modèle fait face à une véritable décision à chaque tour. Demandez « combien font 12 fois 34 ? » et il devrait tendre la main vers la calculatrice. Demandez « quel temps fait-il à Tokyo ? » et il devrait tendre la main vers l'outil météo. Demandez « quelle est la capitale du Japon ? » et — c'est tout aussi important — il ne devrait tendre la main vers *aucun des deux*, parce qu'il le sait déjà.

Rien dans notre code ne fait ce choix. Nous n'inspectons jamais la question de l'utilisateur, nous ne routons jamais les mots « qui sonnent météo » vers l'outil météo. Nous montrons le menu au modèle et le laissons commander. Ce sont les descriptions dans les schémas qui font le vrai travail ici : c'est la seule chose dont dispose le modèle pour décider quel outil convient, et c'est pourquoi les descriptions d'outils s'écrivent comme de petits modes d'emploi plutôt que comme des étiquettes.

Les tests verrouillent exactement cela. Un faux client scénarise le modèle pour qu'il choisisse l'outil météo, et le test vérifie que la fonction météo s'est exécutée et que la calculatrice ne s'est *pas* exécutée — le dispatch envoie la requête à l'outil nommé, pas à celui que nous avons construit en premier. Un autre test scénarise une réponse en texte simple et vérifie que les deux outils étaient proposés mais qu'aucun n'a tourné.

## Vous venez de construire MCP

Un paragraphe de démystification de buzzword, parce que vous l'avez mérité. Vous avez peut-être entendu parler de **MCP** — le Model Context Protocol, le standard qui permet de brancher des outils dans Claude, Cursor et d'autres applications d'IA. Retirez l'acronyme et voici ce qu'est fondamentalement un serveur MCP : le dictionnaire que vous venez de construire, tournant dans son propre processus. La requête `tools/list` du protocole renvoie les schémas du registre — notre ligne `TOOLS`. Sa requête `tools/call` nomme un outil et ses arguments, et le serveur exécute la fonction correspondante — notre `run_tool`. La différence, c'est la tuyauterie : MCP parle JSON-RPC entre processus pour que n'importe quelle application puisse utiliser n'importe quel serveur d'outils, tandis que notre registre vit dans le même fichier Python que la boucle. La forme — un catalogue nommé de schémas, un dispatch par nom — est la même forme. La prochaine fois que vous brancherez un serveur MCP dans une application d'IA, vous saurez qu'il n'y a pas de magie dans la boîte : c'est un range-couverts avec un câble réseau.

## Le voir fonctionner

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-007-multi-tool-dispatch/agent.py
```

Essayez les trois genres de question dans une même session : « combien font 4823 fois 1979 ? », puis « quel temps fait-il à Paris ? », puis « quelle est la capitale de la France ? ». Vous verrez une ligne `[tool] calculator(...)` pour la première, une ligne `[tool] get_weather(...)` pour la deuxième, et aucune ligne d'outil pour la troisième — la même boucle, trois choix différents, aucun fait par notre code.

## Et après

L'agent a maintenant une boîte à outils et y pioche avec discernement. Mais nous avons fait confiance au modèle pour toujours demander poliment : un vrai appel d'outil peut arriver avec des arguments qui ne se lisent pas, un nom que nous n'avons jamais enregistré, ou un outil qui explose en plein vol. Aujourd'hui, deux de ces trois cas font planter la boucle en plein tour. Agent-008 rend le sous-système d'outils honnête face à l'échec — attraper chacun de ces cas, rapporter l'erreur au modèle comme un résultat qu'il peut voir, et le laisser réessayer ou s'excuser au lieu de faire tomber tout l'agent. La solidité avant la substance — la prochaine fois.
