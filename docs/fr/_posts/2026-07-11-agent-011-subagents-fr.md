---
layout: post
title: "Agent-011 : Le Sous agent - Un agent comme outil"
date: 2026-07-11
author: mikamboo
tags: [ia, agents, llm, claude, python, subagents, multi-agent, isolation-de-contexte]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-11-agent-011-subagents %}) | 🇫🇷 Français

Si « agent IA » est le mot à la mode du moment, « multi-agent » est le mot à la mode du mot à la mode. Des équipes d'agents ! Des agents qui se parlent ! Des essaims ! On dirait le moment où la boîte noire devient enfin vraiment magique. Cette étape existe pour vous montrer l'inverse : **un subagent, c'est juste un outil dont l'implémentation se trouve être un autre agent.** Avec tout ce qu'on a déjà construit, ça tient en une quinzaine de lignes.

Voilà où nous en sommes. [Agent-010]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction-fr %}) a refermé l'arc de la mémoire : quand l'historique de conversation devient trop long, l'agent résume son propre passé pour faire de la place. Avant cela, [agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools-fr %}) avait donné à l'agent des outils de savoir — dont une recherche par mots-clés `search` volontairement bête, sur un petit corpus à propos d'une société appelée Nimbus Labs, où l'intelligence vit dans la boucle : chercher, lire le résultat, rechercher avec de meilleurs mots.

Mettez ces deux étapes côte à côte et vous verrez qu'elles sont en tension. Les recherches d'agent-009 sont exactement le genre de chose qui fait gonfler un historique — chaque requête, chaque échec, chaque nouvel essai atterrit dans la liste `messages` — et la compaction d'agent-010 est l'équipe de nettoyage qui éponge tout ça après coup. Cette étape pose une meilleure question : **et si le désordre n'entrait jamais dans l'historique ?**

## Le réflexe de la délégation

Vous le faites déjà. Quand vous demandez à un collègue de « vérifier notre politique de congés », vous ne restez pas derrière son épaule à regarder chaque dossier qu'il ouvre et chaque recherche infructueuse qu'il lance. Il part, il fouille, et il revient avec une phrase : « 25 jours. » Tout le bruit de la recherche est resté dans *sa* tête. Vous n'avez payé que pour la réponse.

C'est toute l'idée du subagent, et elle a un vrai nom : **l'isolation de contexte**. L'agent parent délègue une question. Le subagent brûle autant de tokens qu'il lui en faut — chercher, rater, réessayer — à l'intérieur de son propre historique de conversation privé. Quand il a fini, cet historique est jeté, et une seule chose repasse du côté du parent : la réponse finale, livrée comme un banal `tool_result`, exactement comme la calculatrice livre « 4 ».

La compaction et les subagents sont deux réponses à la même question — *comment garder la fenêtre de contexte petite ?* — prises par les deux bouts. La compaction rétrécit l'historique après coup, et le paie en détails perdus. Un subagent empêche le bruit d'entrer dans l'historique dès le départ : rien à compresser, puisque le parent ne l'a jamais vu.

## Une boucle, enfin nommée

Depuis [agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %}), chaque étape transporte la même boucle interne, réécrite sur place à chaque fois : appeler le modèle, exécuter les outils qu'il demande, lui renvoyer les résultats, recommencer jusqu'à ce qu'il réponde en texte. On ne lui avait jamais donné de nom parce qu'un seul agent en avait besoin. Maintenant ils sont deux — alors elle devient une fonction :

```python
def agent_turn(client, system, registry, messages, output_stream, tag_prefix=""):
    tools = [{"name": name, **entry["schema"]} for name, entry in registry.items()]
    while True:
        message = client.messages.create(
            model=MODEL, max_tokens=1024, system=system,
            tools=tools, messages=messages,
        )
        messages.append({"role": "assistant", "content": message.content})
        if message.stop_reason != "tool_use":
            return next(b.text for b in message.content if b.type == "text")
        # ... exécuter les outils, ajouter les résultats, boucler ...
```

Rien dans le corps n'est nouveau — c'est la boucle d'agent-008, au caractère près, avec le registre passé en paramètre au lieu d'être lu dans une globale. Ce paramètre compte plus qu'il n'en a l'air : pour la première fois dans la série, *deux boîtes à outils différentes existent en même temps*. Le parent reçoit la calculatrice plus le nouvel outil `research`. Le subagent reçoit `search` et rien d'autre — il ne sait pas calculer, et il ne peut pas lancer son propre subagent.

## Les quinze lignes

Une fois la boucle nommée, le subagent est presque une déception :

```python
def research(question, client, output_stream):
    messages = [{"role": "user", "content": question}]
    reply = agent_turn(
        client, RESEARCHER_PROMPT, RESEARCHER_REGISTRY, messages,
        output_stream, tag_prefix="subagent:",
    )
    output_stream.write(
        f"[subagent] worked through {len(messages)} messages; "
        f"returning only the answer\n"
    )
    return reply
```

La première ligne est tout le truc. `messages = [{"role": "user", "content": question}]` — un **historique tout neuf**, qui ne contient rien d'autre que la question. Le subagent n'hérite pas de la conversation du parent ; il part de zéro, travaille dans son propre brouillon, et quand `research` retourne, ce brouillon est ramassé par le garbage collector avec la variable locale. Le parent reçoit une chaîne de caractères.

Le subagent reçoit aussi son propre prompt système, dont la dernière phrase mérite d'être lue deux fois : *« ta réponse est renvoyée à un autre agent, pas montrée à un humain. »* Le public du subagent, ce n'est pas vous — c'est le modèle parent, qui lira la réponse à l'intérieur d'un bloc `tool_result`. Écrire pour un modèle est une vraie compétence, et en voici votre premier aperçu.

L'enregistrement du subagent est le seul endroit où le motif plie légèrement. Tous les outils jusqu'ici étaient de simples fonctions enregistrées au niveau du module. Mais le corps d'un subagent est fait d'appels au modèle, donc il lui faut le `client` — qui n'existe qu'une fois `run` démarré. Alors `run` construit son registre et capture le client dans une closure :

```python
registry = dict(PARENT_REGISTRY)
registry["research"] = {
    "function": lambda question: research(question, client, output_stream),
    "schema": RESEARCH_SCHEMA,
}
```

Du point de vue du harnais de dispatch, `research` est indiscernable de `calculator` : un nom, un schéma, une fonction qui retourne une chaîne. Tout ce qui vient des agents 005 à 008 — le dispatch, la gestion d'erreurs, le renvoi du résultat — s'applique sans changement. C'est la démystification en une phrase : **le parent ne sait pas qu'il a un subagent. Il croit qu'il a un outil.**

## Ce que le multi-agent n'est *pas*

Vous avez peut-être vu l'autre version du multi-agent : des équipes d'agents en jeu de rôle — « l'agent PDG », « l'agent critique » — qui débattent dans une discussion de groupe. Nous ne construisons délibérément pas ça. L'essentiel en est du théâtre de prompts posé sur le mécanisme que vous venez de voir, et la couche qui n'est pas du théâtre, c'est celle-ci : l'isolation de contexte, un agent qui fait le travail bruyant dans un historique privé et rapporte un seul résultat propre. C'est la partie que les systèmes de production utilisent vraiment — c'est ainsi que fonctionnent les subagents de Claude Code — et maintenant, vous l'avez construite.

## Le voir marcher

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-011-subagents/agent.py
```

Demandez « Who founded Nimbus Labs, and what is 25 * 8? ». Regardez le journal : une ligne `[tool] research(...)` quand le parent délègue, puis des lignes `[subagent:tool] search(...)` pendant que le subagent fouille — échecs et nouveaux essais compris — puis `[subagent] worked through N messages; returning only the answer`, et enfin une ligne `[tool] calculator(...)` de retour chez le parent. Deux agents, une conversation, et un seul des deux a vu les recherches.

Les tests verrouillent précisément l'affirmation : le subagent part de rien d'autre que la question (aucun historique hérité), tourne sous son propre prompt système et sa propre boîte à outils, et — l'assertion sur laquelle toute l'étape repose — **aucun de ses tours intermédiaires n'apparaît jamais dans la liste de messages du parent.** Seule la réponse finale y entre, comme un tool_result.

## Et après

L'agent sait désormais se souvenir, utiliser des outils, puiser dans un savoir, survivre aux erreurs, gérer sa propre mémoire et déléguer. À chaque étape, nous l'avons jugé de la même façon : le lancer, le titiller, voir si ça semble juste. C'était acceptable quand l'agent faisait une seule chose ; ça ne l'est plus. Un agent aussi capable peut se tromper de façons qu'un coup d'œil rapide ne détecte pas — et « ça avait l'air de marcher quand j'ai essayé » n'est pas un standard d'ingénierie. Agent-012 affronte cela : les **évals et le traçage** — journaliser ce que l'agent fait réellement à chaque pas, et mesurer, systématiquement et de façon répétable, s'il est réellement bon.
