---
layout: default
title: AI Agent Series
---

# AI Agent Series

🇬🇧 English | [🇫🇷 Français]({{ site.baseurl }}/fr/)

A from-scratch walkthrough of building an AI agent, one stage at a time.
Each post pairs with a folder under [`agents/`](https://github.com/the-ai-coffee/ai-agent-from-scratch/tree/main/agents) in the
repo source, so you can read the explanation and the code side by side.

## Posts

{% for post in site.posts reversed %}{% unless post.categories contains 'fr' %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% endunless %}{% endfor %}
