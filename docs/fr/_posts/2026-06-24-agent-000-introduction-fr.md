---
layout: post
title: "Intro : Construire un agent IA depuis zéro"
date: 2026-06-24
author: mikamboo
tags: [ia, agents, depuis-zero, python, llm]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-06-24-agent-000-introduction %}) | 🇫🇷 Français

« Agent IA » est devenu l'un de ces termes qui paraissent plus compliqués
qu'ils ne le sont. On parle d'agents qui raisonnent, mémorisent, utilisent des
outils — et quelque part en chemin, tout cela finit par ressembler à de la
magie se produisant dans une boîte noire.

J'avais envie d'ouvrir la boîte.

Cette série fait donc la seule chose qui démystifie vraiment les agents :
**les construire depuis zéro.** Pas de frameworks, pas de magie. Juste du
Python simple, et un mécanisme ajouté à la fois, jusqu'à ce qu'une simple
boucle se transforme en un véritable agent capable de mémoriser, de raisonner
et d'utiliser des outils.

**LangGraph, CrewAI, AutoGen — ils enveloppent tous la même petite poignée de
mécanismes :** une boucle, un appel au modèle, un peu de mémoire, quelques
outils et certaines fonctionnalités avancées. C'est tout. Et la meilleure
façon de vraiment comprendre ces pièces, c'est de les écrire soi-même.

Alors écrivons-les nous-mêmes.

## Une seule règle, et pourquoi elle compte

Chaque étape vit dans son propre dossier autonome sous
[`agents/`](https://github.com/the-ai-coffee/ai-agent-from-scratch/tree/main/agents),
et ajoute **exactement un** nouveau mécanisme par-dessus le précédent.

Cette unique contrainte constitue toute la pédagogie. Quand quelque chose finit
par faire tilt — ou casse — il n'y a jamais qu'*une seule* chose nouvelle que
ça puisse être. Vous saurez toujours exactement quelle pièce mobile vous
regardez, parce que vous venez de l'ajouter.

Chaque étape est livrée avec un article comme celui-ci, expliquant *pourquoi*
le mécanisme fonctionne comme il le fait — pas seulement ce que fait le code.
Écrit en supposant que vous avez le code ouvert dans l'autre fenêtre.

## Là où tout ça nous mène

On commence par une boucle qui ne fait rien d'autre qu'un écho. On termine avec
un agent qui possède :

* une mémoire à court terme,
* une mémoire à long terme,
* l'usage d'outils,
* la récupération d'information (RAG),
* la gestion des erreurs,
* la persistance,
* des évaluations (evals).

C'est exactement l'ensemble de fonctionnalités que les grands frameworks vous
vendent — sauf que vous aurez construit chaque pièce à la main. La dernière
étape met votre agent fait maison côte à côte avec LangGraph, CrewAI et
AutoGen, pour que vous puissiez enfin voir ce que ces frameworks font
*réellement* sous le capot. Spoiler : vous reconnaîtrez tout.

## La feuille de route

| # | Étape | Ce que ça ajoute |
|---|-------|---------------|
| 001 | [La boucle d'écho]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop-fr %}) | La simple boucle lire-agir-répéter, sans LLM pour l'instant. |
| 002 | [L'appel au LLM]({{ site.baseurl }}{% post_url 2026-06-26-agent-002-llm-call-fr %}) | L'action devient un unique appel LLM sans état. |
| 003 | conversation-history | Mémoire à court terme : une liste `messages` transportée d'un tour à l'autre. |
| 004 | system-prompt | Une persona/des instructions configurables, séparées des tours. |
| 005 | single-tool-call | Le modèle peut demander un outil ; on l'exécute. |
| 006 | tool-result-loop | Le résultat de l'outil revient au modèle pour une réponse finale. |
| 007 | RAG | Mémoire à long terme : récupération sur un petit vector store, branchée comme un outil. |
| 008 | multi-tool-dispatch | Un vrai registre : le modèle choisit parmi plusieurs outils, ou aucun. |
| 009 | malformed-tool-call-handling | JSON invalide, outils inconnus, exceptions levées — gérés, et non fatals. |
| 010 | persistent-session | L'historique de conversation survit à un redémarrage du processus. |
| 011 | evals-and-tracing | Journalisation pensée-vs-action, plus un harnais d'évaluation scripté. |
| 012 | capstone | Tout combiné, plus une comparaison avec les grands frameworks. |

L'argumentaire de conception complet pour chaque étape — pourquoi elle est
cadrée ainsi, ce qu'elle teste, et comment elle se connecte aux étapes
voisines — se trouve dans
[ROADMAP.md](https://github.com/the-ai-coffee/ai-agent-from-scratch/blob/main/ROADMAP.md).

## Commencez ici

Tout commence là où commence chaque agent, dépouillé de tout le reste : une
boucle.
[Agent-001 construit la boucle]({{ site.baseurl }}{% post_url 2026-06-25-agent-001-echo-loop-fr %})
que tout le reste prolonge — pas de modèle, pas d'outils, juste le battement de
cœur. On se voit là-bas.
