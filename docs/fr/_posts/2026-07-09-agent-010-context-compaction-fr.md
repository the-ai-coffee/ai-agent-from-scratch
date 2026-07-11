---
layout: post
title: "Agent-010 : Un agent qui gère le Contexte"
date: 2026-07-09
author: mikamboo
tags: [ia, agents, llm, claude, python, fenetre-de-contexte, compaction, memoire]
---

[🇬🇧 English]({{ site.baseurl }}{% post_url 2026-07-09-agent-010-context-compaction %}) | 🇫🇷 Français

Dans [agent-003]({{ site.baseurl }}{% post_url 2026-06-27-agent-003-conversation-history-fr %}), nous avons donné une mémoire à l'agent : une liste `messages` qui garde chaque tour de la conversation et qu'on renvoie au modèle à chaque appel. C'était la chose la plus simple qui pouvait marcher, et elle nous a portés à travers sept étapes depuis. Mais elle a un défaut qu'on a poliment ignoré, et chaque étape l'a aggravé. Les appels d'outils ([agent-005]({{ site.baseurl }}{% post_url 2026-06-29-agent-005-single-tool-call-fr %})), les résultats d'outils rebouclés ([agent-006]({{ site.baseurl }}{% post_url 2026-06-30-agent-006-tool-result-loop-fr %})), les rapports d'erreurs ([agent-008]({{ site.baseurl }}{% post_url 2026-07-02-agent-008-malformed-tool-call-handling-fr %})), les chunks de savoir récupérés ([agent-009]({{ site.baseurl }}{% post_url 2026-07-03-agent-009-knowledge-tools-fr %})) — tout s'empile sur cette unique liste, et *rien n'en sort jamais*.

C'est un problème, parce que la mémoire du modèle a une limite. La **fenêtre de contexte** — la quantité totale de texte qu'un modèle peut considérer en un appel — est grande (des centaines de milliers de tokens pour les modèles Claude actuels) mais finie. Parlez assez longtemps, lancez assez d'outils, et un jour l'historique ne tiendra tout simplement plus. L'agent ne se dégrade pas en douceur ; l'API refuse l'appel, point. Notre agent, tel que construit, a une date de péremption qui se compte en tours.

Cette étape ajoute la parade standard, la même que les agents de production comme Claude Code utilisent : la **compaction**. Et il y a quelque chose de plaisamment récursif là-dedans. Notre agent a exactement une compétence — appeler le modèle. Alors quand sa mémoire déborde, il tourne cette compétence sur *lui-même* : il tend ses propres vieux tours au modèle, demande un résumé, et les remplace par celui-ci. L'agent oublie exprès, pour pouvoir continuer.

## Mesurer le problème

Avant de compacter un historique, il faut le mesurer — et le nôtre n'est pas uniforme. Après dix étapes, il contient trois formes de messages : du texte utilisateur (une simple chaîne), du contenu assistant (une liste de blocs de texte et d'appels d'outils), et des résultats d'outils (une liste de dictionnaires). `render_message` aplatit n'importe laquelle en texte lisible :

```python
def render_message(message):
    content = message["content"]
    if isinstance(content, str):
        return f"{message['role']}: {content}"
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(f"tool result: {block['content']}")
        elif block.type == "text":
            parts.append(f"{message['role']}: {block.text}")
        elif block.type == "tool_use":
            parts.append(f"{message['role']} called {block.name}({block.input})")
    return "\n".join(parts)
```

Cette fonction fait double emploi : c'est avec elle qu'on *mesure* l'historique (`history_size` additionne juste les longueurs), et c'est avec elle qu'on écrira la transcription que le modèle résume. Les vrais systèmes comptent des tokens face à la vraie fenêtre ; nous comptons des caractères face à un budget minuscule — `CONTEXT_LIMIT = 1200` — pour que vous puissiez voir la compaction se déclencher en trois tours plutôt qu'en trois heures. Le mécanisme est identique ; seul le mètre-étalon est en format jouet.

## Où l'on peut couper sans danger

Voici la partie subtile, celle qui vaut à cette étape sa place dans la série. On ne peut pas tronçonner l'historique n'importe où. Depuis agent-006, un appel d'outil et son résultat sont les deux moitiés d'une même pensée, étalées sur deux messages : le bloc `tool_use` de l'assistant, puis un message de rôle utilisateur portant le `tool_result`. Coupez entre les deux et vous avez orphelin une requête de sa réponse — l'API rejette l'historique tout net.

La compaction respecte donc deux frontières :

- **Quand :** seulement *entre* les tours utilisateur — en haut de la boucle externe, après que l'échange précédent est entièrement fini et avant que la nouvelle question n'entre dans l'historique. À cet instant, aucun appel d'outil n'attend son résultat.
- **Où :** seulement au début d'un *échange* — la ligne tapée par l'utilisateur plus tout ce que l'agent a fait en réponse. Trouver ces débuts demande une distinction soigneuse : un message de rôle utilisateur n'est un vrai tour utilisateur que si son contenu est une chaîne. Si c'est une liste, ce sont des résultats d'outils, en plein échange — pas un endroit où couper.

```python
def exchange_starts(messages):
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], str)
    ]
```

C'est un schéma à retenir bien au-delà de ce jouet : gérer la mémoire d'un agent est une affaire de *structure*, pas seulement de taille. L'historique n'est pas un ruban qu'on raccourcit à la bonne longueur ; c'est une chaîne de requêtes et de réponses appariées, et on ne peut couper qu'aux articulations.

## Plier le passé

Avec la mesure et les frontières en place, `compact` lui-même est court. Si l'historique tient, ne rien faire. Si tout ce qu'il contient est récent, ne rien faire — aucun intérêt à résumer l'échange sur lequel on est en train de s'appuyer. Sinon : rendre les vieux échanges en transcription, demander un résumé au modèle, et reconstruire l'historique comme *le résumé d'abord, les tours récents tels quels ensuite* :

```python
transcript = "\n".join(render_message(m) for m in messages[:split])
response = client.messages.create(
    model=MODEL,
    max_tokens=512,
    system=SUMMARY_PROMPT,
    messages=[{"role": "user", "content": transcript}],
)
summary = next(b.text for b in response.content if b.type == "text")

return [
    {"role": "user", "content": f"[Conversation summary: {summary}]"},
    {"role": "assistant", "content": "Understood. Continuing from that summary."},
] + messages[split:]
```

Deux détails méritent un second regard. D'abord, le résumé rentre dans l'historique comme un petit échange complet à part entière — un tour utilisateur portant le résumé, plus un accusé de réception d'une ligne côté assistant — pour que les rôles continuent d'alterner exactement comme l'API l'attend. Ensuite, le `SUMMARY_PROMPT` ne dit pas juste « résume » : il dit *garde chaque fait, chiffre, nom et décision dont les tours suivants pourraient avoir besoin*. Cette instruction est tout l'enjeu, parce que —

## — la compaction est avec perte, et c'est le marché

Un résumé est plus petit que ce qu'il résume précisément parce qu'il laisse des choses de côté. Ce que le résumé ne mentionne pas, l'agent l'a véritablement oublié : ce n'est plus dans la liste `messages`, et le modèle ne peut pas voir ce qu'on ne lui envoie pas. La compaction n'est pas un moyen d'avoir une mémoire infinie. C'est un *échange* : le rappel parfait de tout, jusqu'à mourir contre le mur — ou le rappel flou du passé lointain et le rappel parfait du passé récent, pour toujours.

C'est pourquoi les tours récents sont gardés tels quels (`KEEP_RECENT = 1` échange dans notre cas ; les vrais systèmes en gardent plus). Le contexte le plus frais est celui qui a le plus de chances de compter pour la toute prochaine réponse, c'est donc le dernier qu'on voudrait rendre flou. Le passé se compresse ; le présent reste net. Si cela ressemble au fonctionnement de votre propre mémoire d'une longue réunion, ce n'est pas une coïncidence — c'est le même problème d'ingénierie.

Et dans la boucle, le changement tient en une ligne, là où ça compte :

```python
messages = compact(messages, client, output_stream, context_limit)
messages.append({"role": "user", "content": line})
```

Tout le reste — les outils, le registre, le harnais d'erreurs, la boucle interne — est celui d'agent-008, au caractère près. Comme dans agent-009, la récompense d'avoir construit soigneusement, c'est que la capacité suivante s'emboîte sans déranger le reste.

## Le voir tourner

```bash
export ANTHROPIC_API_KEY=<votre clé>
python agents/agent-010-context-compaction/agent.py
```

Posez deux questions bavardes — « tell me about the Eiffel Tower in detail », puis « and the Louvre? » — puis une troisième. Avant la troisième réponse, vous verrez une ligne comme `[compact] folded 4 messages into a summary` : l'agent vient de résumer son propre passé. Demandez-lui maintenant quelque chose sur la *première* réponse. Si le résumé a gardé le fait, il répond depuis le résumé ; sinon, l'agent a honnêtement oublié — la perte d'information, en direct.

Les tests verrouillent chaque pièce : le rendu couvre les trois formes de messages, un historique court passe sans être touché, un message de résultat d'outil n'est jamais pris pour une frontière d'échange, et un scénario scripté de trois tours montre le plus vieil échange se plier en résumé pendant que le plus récent survit tel quel.

## Et après

L'arc de la mémoire ouvert dans agent-003 se referme ici : l'agent sait désormais se souvenir, s'en servir, et — quand il le faut — condenser son propre passé. En chemin, nous avons jugé chaque étape de la même façon : la lancer, la titiller, voir si ça semble juste. C'était acceptable quand l'agent faisait une seule chose. Ça ne l'est plus — un agent avec des outils, du savoir et de la mémoire peut se tromper de façons qu'un coup d'œil rapide ne détecte pas, et « ça avait l'air de marcher quand j'ai essayé » n'est pas un standard d'ingénierie. Agent-011 affronte cela : les **évals** — comment mesurer, systématiquement et de façon répétable, si votre agent est réellement bon.
