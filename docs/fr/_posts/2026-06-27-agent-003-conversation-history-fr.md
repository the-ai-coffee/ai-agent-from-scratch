---
layout: post
title: "Agent-003 : L'historique de conversation"
date: 2026-06-27
author: mikamboo
tags: [ia, agents, llm, claude, python, memoire, conversation]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history %}) | 🇫🇷 Français

[Agent-002]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call-fr %}) a donné un vrai cerveau à notre boucle : au lieu de renvoyer votre ligne en écho, il l'envoyait à Claude et affichait la réponse. Mais cet agent n'avait aucune mémoire. Demandez-lui « quelle est la capitale de la France ? », puis enchaînez avec « et sa population ? », et il n'aurait aucune idée de ce à quoi « sa » renvoie. Chaque ligne atterrissait dans une pièce vide. Cette étape corrige cela — l'agent commence à se souvenir de la conversation.

## La seule chose qui change : une liste qui grandit

Imaginez l'agent précédent comme une personne sans mémoire à court terme. Chaque fois que vous parlez, elle répond parfaitement — puis oublie tout l'échange à l'instant même où il se termine. Pour tenir une véritable conversation avec quelqu'un, il faut retenir ce qui a déjà été dit. Ce « retenir », c'est tout ce qu'est la mémoire ici, et on la construit avec une seule liste Python.

[`agents/agent-003-conversation-history/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-003-conversation-history/agent.py)
garde la même boucle lire-agir-répéter, avec un nouvel ingrédient :

```python
def run(input_stream, output_stream, client=None):
    client = client or Anthropic()
    messages = []

    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        messages.append({"role": "user", "content": line})
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
        )
        reply = message.content[0].text
        messages.append({"role": "assistant", "content": reply})
        output_stream.write(f"Agent> {reply}\n")
```

Tout le changement tient dans la liste `messages` et les deux appels `append` autour de l'appel au LLM :

- `messages = []` est créée **une seule fois**, avant la boucle — elle survit donc d'un tour à l'autre.
- Avant chaque appel, on ajoute la nouvelle ligne de l'utilisateur (`{"role": "user", ...}`).
- Après chaque réponse, on ajoute ce que l'agent a dit (`{"role": "assistant", ...}`).
- Et surtout, on envoie la **liste entière** à Claude à chaque tour, pas seulement la dernière ligne.

Comparez avec agent-002, qui envoyait `messages=[{"role": "user", "content": line}]` — une liste flambant neuve d'un seul élément, construite puis jetée à chaque tour. Cette unique ligne était tout l'univers de l'agent. Ici, la liste ne fait que grandir.

## Pourquoi envoyer tout l'historique à chaque fois ?

C'est la partie qui surprend. Le modèle lui-même ne se souvient de rien entre deux appels — chaque appel `messages.create` est indépendant, un nouveau départ sur les serveurs d'Anthropic. Alors comment l'agent peut-il « se souvenir » ?

L'astuce, c'est que **c'est nous** qui nous souvenons, et que nous re-racontons toute l'histoire au modèle à chaque tour. C'est moins comme parler à un ami qui se rappelle votre dernière discussion, et plus comme travailler avec un consultant brillant frappé d'amnésie totale : avant chaque question, vous lui tendez la transcription complète de tout ce qui a été dit, il la lit, répond, et oublie de nouveau. Comme la transcription ne cesse de s'allonger, ses réponses restent parfaitement dans le contexte — même si sa mémoire est effacée à chaque fois.

Ainsi, quand vous demandez « et sa population ? », la liste que nous envoyons contient déjà « quelle est la capitale de la France ? » et la réponse « Paris ». Le modèle lit cet historique et comprend que « sa » désigne Paris. L'intelligence n'a pas changé entre agent-002 et agent-003 ; ce qui a changé, c'est le *contexte* qu'on lui fournit.

Les deux étiquettes `role` sont la façon dont le modèle distingue les interlocuteurs : `user`, c'est vous, `assistant`, ce sont les réponses passées de l'agent. Inclure les réponses antérieures de l'assistant compte autant qu'inclure les vôtres — c'est ainsi que le modèle sait ce qu'il s'est déjà engagé à dire.

## Ce que cela coûte

La mémoire n'est pas gratuite. Comme on renvoie toute la conversation à chaque tour, chaque appel transporte un peu plus de texte que le précédent. Une longue discussion signifie de plus en plus de mots à chaque requête — ce qui coûte davantage et finit par buter sur la limite de contexte du modèle. Les vrais agents gèrent cela en élaguant ou en résumant les anciens tours, mais pour l'instant la simple liste qui grandit est la manière la plus claire de voir comment fonctionne la mémoire. Nous reviendrons sur ces limites plus tard.

## L'exécuter

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-003-conversation-history/agent.py
```

Essayez un échange en deux temps : posez une question, puis enchaînez avec une question qui n'a de sens que s'il s'est souvenu de la première réponse. Appuyez sur Entrée sur une ligne vide pour arrêter.

## Et après

Notre agent peut désormais tenir une conversation, mais il n'a aucun caractère propre — aucune instruction permanente sur qui il est ni sur la manière dont il doit se comporter. [Agent-004]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt-fr %}) lui donne un **prompt système** : une persona configurable, distincte de la conversation, qui façonne chaque réponse.
