---
layout: post
title: "Agent-012 : Évals et traçage - La confiance n'exclut pas le contrôle"
date: 2026-07-11
author: mikamboo
tags: [ia, agents, llm, claude, python, evals, tracage, observabilite, tests]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-11-agent-012-evals-and-tracing %}) | 🇫🇷 Français

Notre agent a grandi. Depuis [agent-011]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents-fr %}), il se souvient, utilise des outils, puise dans un savoir, survit aux erreurs, gère sa propre mémoire et délègue le travail bruyant à un subagent. Et à chacune de ces onze étapes, nous l'avons jugé de la même façon : le lancer, le titiller, hocher la tête si ça semble juste. C'était acceptable quand l'agent répétait des lignes en écho. Ça ne l'est plus — parce que « ça avait l'air de marcher quand j'ai essayé » n'est pas un standard d'ingénierie, et qu'un agent aussi capable peut se tromper de façons qu'un coup d'œil rapide ne détectera jamais.

Cette étape n'ajoute aucune capacité nouvelle à l'agent. Elle ajoute deux choses *autour* de l'agent, qui répondent à la même question par deux côtés — **que fait réellement l'agent, et est-il réellement bon ?**

- **Le traçage** : un registre continu de chaque décision du modèle, avec son étiquette de prix.
- **Les évals** : des conversations scénarisées aux propriétés attendues, notées automatiquement.

## On ne répare pas ce qu'on ne voit pas

Imaginez une course en taxi où vous ne voyez que le montant à l'arrivée. Quarante euros — était-ce un trajet honnête, ou le chauffeur a-t-il fait trois fois le tour du pâté de maisons ? Sans l'itinéraire, impossible de discuter. Jusqu'ici, notre agent était ce taxi : on tapait une question, une réponse sortait, et tout ce qui se passait entre les deux — combien de fois le modèle a été appelé, quels outils il a saisis, combien de tokens chaque appel a brûlés — était invisible.

Le traçage, c'est la carte du trajet. À partir de cette étape, chaque appel au modèle laisse une ligne dans le journal :

```
[trace] stop=tool_use tools=calculator tokens=312+47 cost=$0.000359
[trace] stop=end_turn tools=- tokens=406+18 cost=$0.000496
```

Lisez-en une à voix haute et c'est une phrase complète sur une décision : *le modèle s'est arrêté parce qu'il voulait un outil ; l'outil était la calculatrice ; l'appel a lu 312 tokens et en a écrit 47 ; ça a coûté un trentième de centime.* La seconde ligne est l'appel de conclusion : pas d'outil, juste la réponse finale.

Il n'y a aucun framework de traçage derrière — c'est le même `output_stream.write` que nous utilisons depuis [agent-001]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop-fr %}), alimenté par deux champs que l'API retourne à chaque réponse (`usage.input_tokens` et `usage.output_tokens`) et deux constantes :

```python
PRICE_PER_MTOK_INPUT = 1.00   # dollars par million de tokens d'entrée (Haiku 4.5)
PRICE_PER_MTOK_OUTPUT = 5.00  # dollars par million de tokens de sortie

def cost_of(usage):
    return (usage.input_tokens * PRICE_PER_MTOK_INPUT
            + usage.output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000
```

Ces constantes comptent plus qu'il n'y paraît. Le token est l'unité native de l'agent, mais personne ne budgète en tokens. Dès qu'une trace parle en *dollars*, deux étapes précédentes cessent d'être abstraites : la compaction d'[agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction-fr %}), c'était « garder l'historique petit » — maintenant vous pouvez regarder `tokens=` grimper tour après tour à mesure que l'historique est renvoyé, et voir exactement ce que la compaction économise. Et le subagent d'agent-011 fait toujours ses recherches dans un historique privé — mais ses appels apparaissent dans la trace, marqués `[subagent:trace]`, parce que l'isolation cache les tokens au *contexte du parent*, pas à la *facture*.

Chaque ligne de trace atterrit aussi dans une simple liste Python, sous forme de dict — mêmes faits, autre public. La ligne de journal est pour les humains devant le terminal. La liste est pour les programmes. Et un programme en particulier constitue toute la seconde moitié de cette étape.

## Deux sortes de tests

Chaque étape jusqu'ici a livré des tests unitaires, et ils partagent tous la même astuce : un faux client qui retourne des réponses scénarisées. Ces tests vérifient *notre* code — la boucle se termine, le dispatch survit à un appel malformé, l'historique du subagent reste privé. Ils sont déterministes, et ils ne posent jamais la seule question qui compte désormais : **face à un vrai modèle et un vrai prompt, l'agent se comporte-t-il bien ?**

C'est une question d'une autre nature. « La boucle se termine-t-elle ? » a une réponse démontrable. « Le modèle saisit-il la calculatrice quand on lui demande de l'arithmétique — au lieu de calculer de tête, peut-être faux ? » n'en a pas. Le modèle est un composant probabiliste ; la seule façon de savoir comment il se comporte, c'est de le lancer et de regarder. Un test qui exécute le système et note le comportement porte un nom dans le monde des LLM : une **éval**.

L'analogie automobile : nos tests unitaires sont le contrôle technique — les freins sont boulonnés, les phares bien câblés. Une éval, c'est l'examen de conduite — on met la voiture sur la route et on regarde ce qu'elle *fait*.

## Le harnais

Dans notre harnais, un cas d'éval est un prompt plus les propriétés que son exécution doit satisfaire :

```python
EVAL_CASES = [
    {
        "name": "uses the calculator for arithmetic",
        "prompt": "What is 6 * 7?",
        "checks": [expect_tool_call("calculator"), expect_reply_contains("42")],
    },
    {
        "name": "delegates company questions to the researcher",
        "prompt": "Who founded Nimbus Labs?",
        "checks": [expect_tool_call("research"), expect_reply_contains("Okonkwo")],
    },
    {
        "name": "answers small talk without any tool",
        "prompt": "Hello! How are you today?",
        "checks": [expect_no_tool_calls()],
    },
]
```

Remarquez ce que les checks ne disent *pas*. Pas de phrase attendue, pas de formulation exacte — le modèle tourne ses réponses différemment à chaque exécution, et épingler la chaîne exacte rendrait chaque éval instable par construction. Chaque check affirme au contraire une **propriété** : un outil qui doit apparaître dans la trace, un fait qui doit apparaître dans la réponse. Un check n'est qu'une fonction sur `(trace, reply)` qui retourne `None` quand la propriété tient, et une réclamation lisible quand elle ne tient pas :

```python
def expect_tool_call(name):
    def check(trace, reply):
        called = [t for entry in trace for t in entry["tools"]]
        if name in called:
            return None
        return f"expected a call to {name!r}, saw {called or 'no tool calls'}"
    return check
```

C'est ici que les deux moitiés de l'étape s'emboîtent : **les checks lisent la trace.** Sans la première moitié, « a-t-il appelé la calculatrice ? » serait sans réponse ; avec elle, la question tient en une compréhension de liste. Et quand un cas échoue, les lignes de trace juste au-dessus du verdict montrent exactement ce que le modèle a fait à la place — l'échec arrive avec son diagnostic attaché.

Le runner lui-même est une boucle que vous avez vue onze fois : historique frais, trace fraîche et boîte à outils fraîche par cas (les évals ne doivent pas fuir l'une dans l'autre), exécuter le tour, appliquer les checks, imprimer un verdict avec le prix :

```
[eval] PASS uses the calculator for arithmetic ($0.000855)
[eval] PASS delegates company questions to the researcher ($0.004964)
[eval] PASS answers small talk without any tool ($0.000527)
[eval] 3/3 passed -- total cost $0.006346
```

`run_evals` retourne le nombre d'échecs, si bien qu'un job de CI peut transformer le comportement de l'agent en code de sortie — l'humble mécanisme par lequel « l'agent a régressé » devient un build rouge plutôt qu'un ticket de support. Changez le prompt système, échangez le modèle, réordonnez les outils : lancez les évals et vous saurez en quelques secondes si vous avez cassé quelque chose, et la trace vous dira comment.

## Le voir marcher

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-012-evals-and-tracing/agent.py --eval   # lancer les évals
python agents/agent-012-evals-and-tracing/agent.py          # ou discuter, désormais tracé
```

En mode discussion, chaque échange déroule maintenant ses lignes de trace entre votre question et la réponse, et quitter imprime le total de la session. Posez la question sur Nimbus Labs de l'étape précédente et regardez le registre voir à travers l'isolation du subagent : `[trace]` pour la délégation du parent, `[subagent:trace]` pour chaque recherche privée, chaque ligne chiffrée.

Les tests unitaires, eux, font ce qu'ils ont toujours fait — avec le faux client, ils verrouillent la *mécanique* de façon déterministe : une entrée de trace par appel au modèle, des coûts calculés aux bons tarifs, un check qui échoue produisant une ligne `FAIL` et un retour non nul. La mécanique se teste avec des faux ; le comportement se teste avec des évals. Les deux sortes, chacune pour ce qu'elle sait faire.

## Et après

L'agent est construit — et maintenant il est observable et mesurable, en plus. Il y a treize étapes, cette série promettait qu'« agent IA » ne cache aucune magie, juste une boucle ; chaque mécanisme que vendent les frameworks est depuis sorti de cette boucle, un petit morceau à la fois. Il reste une étape, et elle n'ajoute rien de neuf, à dessein : un **capstone** qui fait tout tourner ensemble — les outils de savoir, un subagent qui délègue, la compaction, le harnais d'évals — puis se retourne vers LangGraph, CrewAI et AutoGen avec des yeux de bâtisseur, pour voir ce que ces frameworks font réellement pour vous.
