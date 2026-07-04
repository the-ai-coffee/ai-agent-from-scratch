---
layout: post
title: "Agent-006 : Fermer la boucle"
date: 2026-06-30
author: mikamboo
tags: [ia, agents, llm, claude, python, outils, function-calling]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop %}) | 🇫🇷 Français

Agent-005 s'est terminé sur un suspense. Nous avions donné une calculatrice à l'agent, nous l'avions vu *décider* de tendre la main vers l'outil, nous avions exécuté l'outil et affiché le nombre brut — puis nous nous étions arrêtés. Le modèle avait demandé un calcul, nous l'avions fait, mais nous n'avions jamais dit au modèle ce qui était revenu. La main s'était tendue et avait saisi quelque chose ; le cerveau n'avait jamais su quoi. Cette étape corrige cela, et ce faisant construit la boucle qui fait d'un agent un agent.

## La moitié qui manquait

Pensez à la façon dont vous utilisez réellement une calculatrice. Vous ne tapez pas simplement `2 + 3`, ne lisez pas « 5 » et ne repartez pas satisfait. Vous lisez le 5, et *ensuite* vous dites quelque chose : « bon, le total fait donc cinq. » Le nombre est une entrée pour votre pensée suivante, pas sa fin.

Agent-005 donnait à l'agent la première moitié de cela — tendre la main vers l'outil — mais pas la seconde. Le résultat ne repartait jamais vers le modèle, si bien que le modèle ne pouvait jamais rien dire *à son sujet*. Tout ce que nous pouvions vous montrer, c'était le `5` nu. Ce qui manquait, ce n'était pas un outil plus gros ou un modèle plus intelligent. C'était un chemin de retour : un moyen de ramener le résultat dans la conversation pour que le modèle puisse y réagir.

## Un aller-retour, décomposé

Voici tout le cycle que construit cette étape, pas à pas :

1. L'utilisateur demande quelque chose.
2. Le modèle répond — mais au lieu d'une phrase, il réclame un outil.
3. Nous exécutons l'outil et obtenons un résultat.
4. Nous renvoyons le résultat au modèle sous la forme d'un nouveau tour.
5. Le modèle, capable désormais de voir le résultat, rédige sa vraie réponse.

Les étapes 1 à 3 sont exactement agent-005. Les étapes 4 et 5 sont nouvelles, et c'est ce que signifie « fermer la boucle ». L'astuce, c'est que l'étape 4 ne termine pas le tour — elle repart vers l'étape 2. Le modèle peut regarder le résultat et décider qu'il a besoin d'*un autre* appel d'outil avant d'être prêt à répondre. Ce n'est donc pas une ligne droite ; c'est une boucle qui continue tant que le modèle ne cesse pas de réclamer des outils.

## La boucle interne

Dans le code, cela se traduit par une boucle *à l'intérieur* de la boucle par ligne :

[`agents/agent-006-tool-result-loop/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-006-tool-result-loop/agent.py)

```python
messages.append({"role": "user", "content": line})

# Boucle interne : rappeler le modèle jusqu'à ce qu'il cesse de réclamer des outils.
while True:
    message = client.messages.create(
        model=MODEL, max_tokens=1024, system=system,
        tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": message.content})

    if message.stop_reason != "tool_use":
        reply = next(b.text for b in message.content if b.type == "text")
        output_stream.write(f"Agent> {reply}\n")
        break

    tool_results = []
    for block in message.content:
        if block.type != "tool_use":
            continue
        result = run_tool(block.name, block.input)
        output_stream.write(f"[tool] {block.name}({block.input}) -> {result}\n")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })
    messages.append({"role": "user", "content": tool_results})
```

La boucle externe (non montrée ici) lit une ligne de l'utilisateur, comme avant. Le `while True` **interne** est la partie nouvelle : il rappelle le modèle encore et encore pour cette *seule* ligne utilisateur, tant que le modèle continue de réclamer des outils. Quand le modèle répond enfin avec des mots — `stop_reason` n'est plus `"tool_use"` — nous affichons la réponse et sortons avec `break`, de retour en attente de la ligne utilisateur suivante.

## Deux choses entrent désormais dans l'historique

Dans agent-005 nous *retirions* délibérément le tour d'outil de la mémoire (`messages.pop()`), parce qu'une requête sans résultat renvoyé casserait l'appel suivant. Maintenant nous faisons l'inverse : nous enregistrons les deux moitiés.

D'abord, la requête du modèle elle-même :

```python
messages.append({"role": "assistant", "content": message.content})
```

Nous stockons la réponse du modèle *telle quelle*, requête d'outil comprise. C'est important : le résultat d'outil que nous sommes sur le point d'ajouter doit pointer vers une requête précise, et cette requête doit se trouver dans l'historique pour que le pointeur ait un sens.

Ensuite, le résultat, emballé en `tool_result` :

```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": result,
})
```

Le `tool_use_id` est le fil qui relie les deux. Quand le modèle a réclamé la calculatrice, sa requête portait un `id`. Notre résultat cite ce même `id` en retour. C'est ainsi que le modèle sait que *cette* réponse appartient à *cette* question — indispensable dès qu'un tour comporte plus d'un appel d'outil, sinon les résultats formeraient un tas sans étiquette. C'est la différence entre « voici 5, voici 20 » et « la chose demandée à 15 h 01 fait 5 ; la chose demandée à 15 h 02 fait 20 ».

## Pourquoi elle peut boucler plus d'une fois

Remarquez que nous ne supposons jamais que le modèle veut exactement un appel d'outil. Après avoir renvoyé un résultat, nous rebouclons directement vers le haut et rappelons le modèle — et il est libre de réclamer *un autre* outil. Peut-être avait-il besoin de calculer un sous-total, de le voir, puis d'en calculer un second. La boucle gère naturellement une chaîne d'appels d'outils de n'importe quelle longueur, car la seule chose qui y met fin est le choix du modèle de répondre avec des mots.

C'est le saut discret de cette étape. Jusqu'ici, une ligne utilisateur signifiait exactement un appel de modèle. Maintenant une seule ligne peut déclencher toute une séquence d'étapes penser-agir-observer avant que l'agent ne parle. Cette séquence — le modèle qui décide, agit, voit ce qui s'est passé, et décide à nouveau — c'est ce que les gens entendent lorsqu'ils appellent quelque chose un « agent » plutôt qu'un chatbot.

## Savoir quand s'arrêter

Une boucle qui décide de sa propre longueur soulève une inquiétude évidente : et si elle ne s'arrêtait jamais ? Ici, la condition d'arrêt est entièrement l'affaire du modèle — il met fin à la boucle en répondant avec du texte au lieu de réclamer un autre outil. En pratique, dès que le modèle a le nombre qu'il lui faut, il répond. Mais « le modèle décide quand s'arrêter » est une promesse à tenir avec prudence : un modèle désorienté *pourrait* continuer d'appeler des outils en rond. Les systèmes réels ajoutent un plafond ferme sur le nombre d'itérations comme filet de sécurité. Nous l'avons laissé de côté ici pour garder la boucle nue, mais il faut savoir que la version brute fait confiance au modèle pour savoir quand il a fini.

## Le voir fonctionner

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-006-tool-result-loop/agent.py
```

Demandez « combien font 4823 fois 1979 ? » et comparez à agent-005. Vous verrez toujours la ligne `[tool]` — l'agent qui tend sa main — mais elle est désormais suivie d'une vraie phrase : le modèle a pris le résultat et a formulé une réponse autour de lui, au lieu de déverser le nombre brut. L'écart de l'étape précédente est comblé.

## Et après

Nous avons maintenant le battement de cœur complet d'un agent : appeler, agir, renvoyer, répondre, recommencer. Mais l'unique outil de notre agent ne fait que de l'arithmétique — il ne peut toujours rien *rechercher*. Agent-007 lui donne une mémoire qu'il peut interroger : une petite base de connaissances, et un outil de recherche branché directement sur la boucle que nous venons de construire. Le chemin de retour du résultat de cette étape est précisément ce qui ramène un fait retrouvé jusqu'au modèle. C'est ainsi qu'un agent cesse de se limiter à ce qui est dans son contexte et commence à aller chercher ce dont il a besoin — la prochaine fois.
