---
layout: post
title: "Agent-001 : La boucle d'écho"
date: 2026-06-25
author: mikamboo
tags: [ia, agents, depuis-zero, python, agent-loop, tests]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop %}) | 🇫🇷 Français

Tout agent, aussi capable soit-il, est construit autour de la même forme :
lire une observation, décider d'une action, recommencer. Avant d'ajouter un
LLM, des outils ou des évaluations, il vaut la peine de construire cette boucle
seule, pour que les étapes suivantes aient quelque chose de concret à étendre.

## Le code

[`agents/agent-001-echo-loop/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-001-echo-loop/agent.py)
implémente la boucle sans aucune dépendance :

```python
def run(input_stream, output_stream):
    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        output_stream.write(f"Agent> {line}\n")
```

- **Observation** : une ligne lue depuis `input_stream`.
- **Action** : réécrire cette ligne vers `output_stream`.
- **Boucle** : la boucle `while`, qui continue jusqu'à un EOF ou une ligne vide.

Passer `input_stream` et `output_stream` (plutôt que de coder en dur
`sys.stdin`/`sys.stdout`) est ce qui rend ceci testable sans lancer de
processus — les tests dans `test_agent.py` passent simplement des objets
`io.StringIO`.

## L'exécuter

```bash
python agents/agent-001-echo-loop/agent.py
```

Tapez une ligne, appuyez sur Entrée, voyez-la renvoyée en écho. Appuyez sur
Entrée sur une ligne vide pour arrêter.

## Et après

[Agent-002]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call-fr %}) remplace l'écho par un véritable appel LLM, en conservant la même forme __lire-agir-répéter__.
