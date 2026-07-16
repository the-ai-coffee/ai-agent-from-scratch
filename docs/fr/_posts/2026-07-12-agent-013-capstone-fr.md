---
layout: post
title: "Agent-013 : Le Capstone - Tout à la fois, rien de neuf"
date: 2026-07-12
author: mikamboo
tags: [ia, agents, llm, claude, python, capstone, langgraph, crewai, autogen, mcp]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-12-agent-013-capstone %}) | 🇫🇷 Français

Il y a treize étapes, cette série a fait une promesse : *« agent IA » ne cache aucune magie. Juste une boucle.* Depuis, nous avons construit, un petit morceau à la fois, tout ce que vendent les frameworks — mémoire, outils, gestion d'erreurs, savoir, contexte auto-géré, délégation multi-agent, observabilité, évals. Cette étape finale tient la promesse en n'ajoutant **rien de neuf**. Elle compose ce qui existe déjà dans un seul fichier, fait tout tourner dans une seule conversation scénarisée, puis — pour la première fois de la série — regarde vers l'extérieur, du côté de LangGraph, CrewAI, AutoGen et MCP, avec des yeux de bâtisseur.

## Un fichier, treize étapes

Ouvrez [`agents/agent-013-capstone/agent.py`](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/agents/agent-013-capstone/agent.py) et lisez-le de haut en bas : chaque morceau porte un numéro d'étape. La boucle lire-agir, c'est [agent-001]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop-fr %}) ; l'appel au modèle, [agent-002]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call-fr %}) ; la liste `messages` qui grandit, [agent-003]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history-fr %}) ; le prompt système, [agent-004]({{ site.baseurl }}{% post_url 2026-06-28-agent-004-system-prompt-fr %}). Les outils arrivent avec [agent-005]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call-fr %}), rebouclent avec [agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %}), se multiplient en registre avec [agent-007]({{ site.baseurl }}{% post_url 2026-07-01-agent-007-multi-tool-dispatch-fr %}), et cessent d'être fragiles avec [agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling-fr %}). L'outil `search` sur le corpus Nimbus Labs, c'est [agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools-fr %}) ; la compaction, [agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction-fr %}) ; le subagent de recherche, [agent-011]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents-fr %}) ; les lignes de trace et le harnais d'évals, [agent-012]({{ site.baseurl }}{% post_url 2026-07-11-agent-012-evals-and-tracing-fr %}).

Le seul travail d'assemblage de cette étape consiste à remettre la compaction d'agent-010 dans la boucle construite par agent-011 et agent-012 — une ligne avant chaque tour utilisateur — avec une petite attention au passage : le résumé de compaction est un appel au modèle comme les autres, il atterrit donc désormais dans la trace sous l'agent `compactor`. Oublier coûte de l'argent aussi, et le registre doit le dire.

## La démo : tout voir se déclencher dans une seule conversation

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-013-capstone/agent.py --demo
```

La démo, ce sont trois questions scénarisées passées dans le REPL ordinaire — pas de chemin de code spécial, juste `run` qui lit un script préparé au lieu d'un clavier, avec une limite de contexte assez petite pour que la compaction se produise sous vos yeux. Voici une vraie exécution, abrégée :

```
User> Who founded Nimbus Labs, and where is its headquarters today?
[trace] stop=tool_use tools=research,research tokens=778+100 cost=$0.001278
[subagent:trace] stop=tool_use tools=search tokens=697+57 cost=$0.000982
[subagent:tool] search({'query': 'founder founded Nimbus Labs'}) -> ...
[subagent] worked through 4 messages; returning only the answer
[tool] research({'question': 'Who founded Nimbus Labs?'}) -> Nimbus Labs was
founded in 2019 by Dara Okonkwo. ...
Agent> Nimbus Labs was founded in 2019 by Dara Okonkwo. The company's
headquarters is currently located in Porto, Portugal ...

User> Their staff get 25 vacation days. If each day is worth $180, what is
the whole allowance worth?
[trace] stop=tool_use tools=calculator tokens=1084+56 cost=$0.001364
[tool] calculator({'expression': '25 * 180'}) -> 4500
Agent> The whole vacation allowance is worth $4,500 (25 days × $180 per day).

User> Last one: what is the company's flagship product, and who is the office dog?
[compactor:trace] stop=end_turn tools=- tokens=224+42 cost=$0.000434
[compact] folded 4 messages into a summary
[trace] stop=tool_use tools=research,research tokens=966+115 cost=$0.001541
[subagent:tool] search({'query': 'flagship product'}) -> ...
Agent> Flagship product: Skyline, a weather-forecasting platform.
Office dog: a corgi named Biscuit.

[trace] session total: 15 model calls, $0.016220
```

Lisez-le comme une checklist. La première question est déléguée au subagent de recherche, qui fouille la base de connaissances dans son historique privé et rend un seul paragraphe propre. La deuxième question part vers la calculatrice — le modèle ne fait pas l'arithmétique de tête. Avant la troisième question, la ligne `compactor` se déclenche : le premier échange est replié en résumé, et la conversation continue sur un historique plus petit. Chacune de ces décisions est tracée et chiffrée, et la session se termine sur son total : quinze appels au modèle, environ un centime et demi.

C'est toute la série en vingt lignes de terminal. `--eval` fonctionne toujours aussi — l'agent qui fait tout cela est le même que le harnais note.

## Et maintenant, ces fameux frameworks

L'introduction avait nommé LangGraph, CrewAI et AutoGen et promis qu'on y reviendrait une fois que vous pourriez les juger en bâtisseur plutôt qu'en client. Ce moment est arrivé : vous avez construit vous-même les mécanismes qu'ils vendent. Traduisons donc leurs brochures.

**LangGraph** vend l'agent comme un *graphe* : des nœuds, des arêtes, et un objet d'état qui circule entre eux. Cet objet d'état, vous l'avez construit — c'est la liste `messages` d'agent-003. Le graphe, vous l'avez construit aussi — c'est la boucle `while True` d'agent-006, où « quel nœud tourne ensuite » est exactement `stop_reason` : `tool_use` part vers les outils, tout le reste sort. Un graphe est une généralisation de notre boucle, et une généralisation honnête — certains workflows branchent vraiment. Mais quand la documentation de LangGraph montre un cycle entre un nœud « agent » et un nœud « tools », vous regardez un schéma d'`agent_turn`.

**CrewAI** vend des *équipes d'agents à rôles* : un chercheur, un rédacteur, un orchestrateur, chacun avec son rôle, ses objectifs et ses outils. Ça aussi, vous l'avez construit. Un « rôle », c'est un prompt système — agent-004. Un membre d'équipe avec ses propres outils, c'est un subagent avec son propre registre — agent-011, où le chercheur avait `search`, le parent `calculator`, et aucun ne pouvait toucher aux outils de l'autre. L'« orchestrateur » qui décide qui travaille ensuite, c'est le modèle parent qui choisit quel outil appeler. Le fossé multi-agent, vu de près, c'est `RESEARCHER_PROMPT` plus une liste `messages` fraîche.

**AutoGen** vend des *conversations entre agents* : des agents qui s'écrivent jusqu'à ce que le travail soit fait. Regardez ce que notre parent et notre subagent échangent réellement — une question descend en `tool_use`, une réponse remonte en `tool_result`. C'est *exactement* deux agents qui s'écrivent ; la conversation est simplement typée. Le chat de groupe avec un manager qui choisit le prochain intervenant, c'est le même mécanisme avec plus de participants.

Rien de tout cela n'est une accusation. Les frameworks ne cachent pas la boucle — ils l'emballent, et l'emballage a une vraie valeur à l'échelle : persistance et reprise (LangGraph sauvegarde une session, un crash ne la perd pas), réessais, streaming, branches parallèles, tableaux de bord, conventions d'équipe pour que cinq ingénieurs structurent leurs agents de la même façon. Voilà ce que vous construiriez ensuite si cette série continuait un an, et les acheter est souvent le bon choix.

Ce que vous ne devriez plus accepter, c'est de les acheter *à l'aveugle*. Le coût d'un framework, c'est que quand l'agent déraille — et agent-012 nous a appris qu'il déraillera — vous déboguez leur boucle au lieu de la vôtre, à travers leurs abstractions, avec leur vocabulaire. Après treize étapes, vous savez ce qu'il y a sous le plancher : une liste `messages`, un registre d'outils, un test sur `stop_reason`, un appel de résumé. Quand l'« AgentExecutor » d'un framework lève une « OutputParserException », vous savez désormais quoi demander : quelle partie d'agent-008 ont-ils ratée ?

## MCP : la pièce à adopter, pas à démystifier

Un sigle mérite le traitement inverse. Le **Model Context Protocol** n'emballe pas la boucle — il standardise la seule interface que cette série n'a cessé de reconstruire à la main : le registre d'outils.

Rappelez-vous l'entrée de registre d'agent-007 : un nom, une description, un `input_schema`, et une fonction à appeler. Chaque outil de cette série — calculatrice, météo, recherche, jusqu'au subagent de recherche — avait exactement cette forme, parce que c'est la forme qu'attend l'API du modèle. MCP prend cette forme et en fait un protocole *entre processus* : un serveur MCP publie des outils (nom, description, schéma), n'importe quel client MCP peut les lister et les appeler. C'est le standard USB des outils — écrits une fois, branchés dans Claude Code, dans votre agent, dans l'agent de n'importe qui.

Voilà pourquoi il gagne une recommandation là où les frameworks gagnaient une traduction : MCP standardise la partie ennuyeuse pour qu'elle se partage, au lieu d'abstraire la partie intéressante pour qu'elle se vende. Si vous prolongez l'agent construit ici, la suite honnête n'est pas d'adopter un framework — c'est d'apprendre à `run_tool` à parler MCP, pour que votre registre se remplisse tout seul depuis des serveurs écrits par d'autres.

## La fin de la boucle

Cette série s'était donné une question : qu'y a-t-il *réellement* dans un agent IA ? Voici la réponse entière, une ligne par étape. Une boucle qui lit et agit (001). Un appel au modèle dedans (002). Une liste qui se souvient (003) sous des instructions qui persistent (004). Des outils que le modèle peut demander (005) et dont il apprend (006), plusieurs (007), sans casser (008). Un savoir atteint en cherchant, pas en mémorisant (009). Une mémoire qui se résume elle-même plutôt que de déborder (010). Une délégation vers une seconde boucle à liste privée (011). Un registre de chaque décision, et des tests qui notent le comportement plutôt que le code (012).

Aucune magie. Juste une boucle — et maintenant, elle est à vous. Clonez le dépôt, lancez la démo, cassez les évals, ajoutez un outil, écrivez un client MCP, lancez un deuxième subagent. La prochaine étape n'est pas dans cette série. C'est ce que vous construirez.
