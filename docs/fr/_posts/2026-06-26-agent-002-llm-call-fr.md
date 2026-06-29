---
layout: post
title: "Agent-002 : L'appel au LLM"
date: 2026-06-26
author: mikamboo
tags: [ia, agents, llm, claude, python, anthropic-sdk]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call %}) | 🇫🇷 Français

Agent-001 a construit la boucle lire-agir-répéter avec l'action la plus simple possible : renvoyer la ligne en écho. Cette étape garde cette boucle entièrement intacte et y substitue la seule chose qui change entre un écho et un agent — l'action est désormais un appel à un LLM, par exemple Claude AI.

## Le code

[`agents/agent-002-llm-call/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-002-llm-call/agent.py)
ressemble presque à l'identique à la boucle d'agent-001 :

```python
def run(input_stream, output_stream, client=None):
    client = client or Anthropic()

    while True:
        output_stream.write("User> ")
        output_stream.flush()
        line = input_stream.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if not line:
            break
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": line}],
        )
        output_stream.write(f"Agent> {message.content[0].text}\n")
```

- **Observation** : toujours une ligne lue depuis `input_stream`.
- **Action** : un appel `messages.create` à Claude au lieu d'un écho.
- **Boucle** : inchangée par rapport à agent-001 — même invite, même
  terminaison sur EOF/ligne vide.

Chaque ligne est envoyée seule, sans aucune mémoire des lignes précédentes de
la conversation. Il n'y a pas encore d'historique de messages à maintenir, donc
l'agent traite chaque ligne comme une invite fraîche et indépendante — de la
même manière qu'agent-001 traitait chaque ligne comme un écho indépendant.

## Pourquoi `client` est un paramètre

`client` vaut par défaut une véritable instance `Anthropic()` (qui lit
`ANTHROPIC_API_KEY` depuis l'environnement), mais on peut la remplacer. C'est la même idée que de passer `input_stream`/`output_stream` au lieu de coder en dur `sys.stdin`/`sys.stdout` : cela permet à `test_agent.py` de substituer un faux client qui renvoie une réponse préenregistrée, pour que la suite de tests s'exécute sans accès réseau ni clé d'API.

## L'exécuter

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-002-llm-call/agent.py
```

Tapez une ligne, appuyez sur Entrée, voyez la réponse de Claude. Appuyez sur Entrée sur une ligne vide pour arrêter.

## Et après

Cette étape est sans état — l'agent n'a aucune idée de ce que vous avez dit à la ligne précédente. Agent-003 commencera à transporter l'historique de conversation entre les tours, et les étapes ultérieures ajouteront des outils pour que l'agent puisse agir sur plus que du simple texte.
