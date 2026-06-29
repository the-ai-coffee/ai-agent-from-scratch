---
layout: default
title: Série Agent IA
permalink: /fr/
---

# Série Agent IA

[🇬🇧 English]({{ site.baseurl }}/) | 🇫🇷 Français

Un guide pas à pas pour construire un agent IA depuis zéro, une étape à la fois.
Chaque article est associé à un dossier sous [`agents/`](https://github.com/the-ai-coffee/ai-agent-from-scratch/tree/main/agents)
dans le code source, afin que vous puissiez lire l'explication et le code côte à côte.

## Articles

{% for post in site.categories.fr reversed %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
