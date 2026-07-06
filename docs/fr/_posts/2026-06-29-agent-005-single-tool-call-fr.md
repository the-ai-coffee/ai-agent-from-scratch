---
layout: post
title: "Agent-005 : Le premier outil"
date: 2026-06-29
author: mikamboo
tags: [ia, agents, llm, claude, python, outils, function-calling]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call %}) | 🇫🇷 Français

À l'étape [agent-004]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt-fr %}), notre agent avait une mémoire et une persona, mais il ne faisait toujours que *parler*. Demandez-lui « combien font 4823 fois 1979 ? » et il produisait un nombre avec assurance — qui pouvait être faux, car un modèle de langage prédit du texte, il ne calcule pas. Il ne pouvait rien rechercher, rien exécuter, ni toucher au monde au-delà de ses propres mots. Cette étape ouvre une brèche. Nous donnons à l'agent son premier **outil**.

## Un cerveau qui reçoit sa première main

Jusqu'ici notre agent était un cerveau dans un bocal — il sait raisonner et converser, mais il n'a pas de mains. Un outil est une main. Concrètement, un outil n'est rien d'autre que du code que *nous* écrivons et que l'agent est autorisé à nous demander d'exécuter pour lui.

Le mot *demander* est l'essentiel. Le modèle ne tend jamais la main pour exécuter du code lui-même ; il ne le peut pas. Ce que nous faisons, c'est lui présenter un menu d'outils qu'il a le droit de réclamer. Quand il décide qu'il en a besoin, il ne répond pas par du texte — il répond par une requête structurée : « exécute la calculatrice avec l'expression `4823 * 1979`, s'il te plaît. » Le modèle apporte le *jugement* sur le moment où un outil est nécessaire ; notre code apporte l'*action*.

## Décrire l'outil

Avant que le modèle puisse réclamer un outil, il doit savoir que l'outil existe et ce qu'il attend. C'est une simple description que nous envoyons à chaque requête :

[`agents/agent-005-single-tool-call/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-005-single-tool-call/agent.py)

```python
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the result. "
            "Supports +, -, *, /, and parentheses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate, e.g. '2 + 2 * 3'.",
                }
            },
            "required": ["expression"],
        },
    }
]
```

C'est une étiquette, une phrase d'explication, et une description des entrées. Le modèle lit ce menu et décide, tout seul, si une question donnée fait partie de celles que la calculatrice pourrait résoudre. Ces champs `description` ne sont pas décoratifs — c'est ainsi que le modèle sait à quoi sert l'outil et comment remplir ses arguments. Une description vague vous donne un outil que le modèle utilise mal, ou pas du tout.

## Comment savoir que le modèle veut l'outil

Nous envoyons la conversation avec le menu des outils (`tools=TOOLS`), puis nous regardons *comment* le modèle a choisi de répondre :

```python
message = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    system=system,
    tools=TOOLS,
    messages=messages,
)

if message.stop_reason == "tool_use":
    tool_use = next(b for b in message.content if b.type == "tool_use")
    result = run_tool(tool_use.name, tool_use.input)
    output_stream.write(f"[tool] {tool_use.name}({tool_use.input}) -> {result}\n")
    output_stream.write(f"Agent> {result}\n")
    messages.pop()
    continue

reply = message.content[0].text
# ... réponse texte normale ...
```

Le signal est `stop_reason`. C'est le modèle qui nous dit *pourquoi* il a cessé de parler. D'ordinaire il s'arrête parce qu'il a fini sa phrase — une réponse texte normale. Mais quand il s'arrête avec `"tool_use"`, il nous dit autre chose : « je n'ai pas terminé ; j'ai besoin que tu exécutes quelque chose d'abord. » Nous extrayons alors la requête de la réponse (le bloc `tool_use`, qui porte le nom de l'outil et les arguments remplis par le modèle), nous la passons à `run_tool`, et nous exécutons le code correspondant.

Cette ligne `[tool] ...` que nous affichons, c'est l'agent qui tend sa main, rendu visible. Cela vaut la peine de l'afficher, car le plus intéressant chez un agent n'est pas sa réponse finale — c'est de le voir *décider* d'agir.

## Le suspense délibéré

Regardez de près ce qui se passe après l'exécution de l'outil : nous affichons le résultat brut et… nous nous arrêtons. Nous ne renvoyons pas la réponse au modèle. Le mieux que l'agent puisse faire, c'est donc de vous tendre le nombre nu :

```
User> what is 2 + 3?
[tool] calculator({'expression': '2 + 3'}) -> 5
Agent> 5
```

Il ne peut pas encore dire « Cela fait 5. » — parce que le modèle n'a jamais vu le `5`. Il a demandé le calcul, nous l'avons fait, mais la conversation a continué sans jamais dire au modèle ce qui était revenu. La main s'est tendue et a saisi quelque chose ; le cerveau n'a jamais su quoi.

Ce n'est pas un oubli — c'est la couture entre cette étape et la suivante. Nous nous arrêtons délibérément à mi-chemin du cycle d'utilisation de l'outil pour que vous puissiez voir les deux moitiés séparément : *cette* étape, c'est « le modèle demande, nous exécutons » ; l'étape suivante, c'est « le résultat repart, et le modèle parle ». Les séparer rend chaque moitié lisible avant qu'elles ne fusionnent en la boucle qui définit un agent.

Il y a aussi une raison pratique au fait que nous retirons le tour de la mémoire (`messages.pop()`). Une requête d'outil à laquelle aucun résultat n'est renvoyé est un échange inachevé — la laisser dans l'historique casserait l'appel suivant. Puisque le modèle ne fait pas vraiment partie de cet échange pour l'instant, le plus propre est de ne pas l'enregistrer. À l'étape [agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %}), l'enregistrer correctement est précisément ce qui permet au modèle de poursuivre.

## Une note sur la sécurité

Notre calculatrice n'utilise pas le `eval` de Python — cela permettrait à une expression bien conçue d'exécuter du code arbitraire. À la place, elle analyse l'expression et la parcourt, n'autorisant que les nombres et une poignée d'opérateurs arithmétiques. C'est un détail ici, mais une habitude à prendre tôt : **un outil exécute du vrai code sur votre machine.** Dès l'instant où vous laissez un modèle déclencher des actions, vous êtes responsable de vous assurer que ces actions ne peuvent faire que ce que vous voulez. (C'est aussi là que l'inquiétude liée à l'injection de prompt d'agent-004 prend du mordant — un outil transforme la persuasion en conséquences.)

## L'exécuter

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-005-single-tool-call/agent.py
```

Demandez-lui quelque chose d'arithmétique — « combien font 4823 fois 1979 ? » — et observez la ligne `[tool]` apparaître, puis le résultat brut. Puis demandez quelque chose d'ordinaire, comme « quelle est la capitale de la France ? », et remarquez qu'il *n'appelle pas* l'outil : le modèle sait que la calculatrice ne peut pas l'aider ici. Appuyez sur Entrée sur une ligne vide pour arrêter.

## Et après

Pour l'instant, l'agent peut agir, mais il ne peut pas parler de ce qu'il a fait — le résultat de l'outil n'atteint jamais le modèle. [Agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %}) ferme cette boucle : nous renvoyons le résultat dans la conversation et laissons le modèle rédiger une vraie réponse autour de lui. Ce simple aller-retour — demander, exécuter, renvoyer, répondre — est le battement de cœur de tout agent, et c'est la prochaine chose que nous construisons.
