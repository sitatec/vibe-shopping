---
title: Vibe Shopping
emoji: 🛍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.33.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI powered clickless shopping experience with virtual try-on
tags:
  - agent-demo-track
---

# Vibe Shopping

[![](https://github.com/user-attachments/assets/a445d974-7ee0-411b-a33b-e865c1c9e7bd)](https://www.youtube.com/watch?v=tSkny9_AjQs)

Vibe Shopping is an AI-powered clickless shopping experience that allows users to virtually try on clothes and accessories.

It uses Qwen-2.5-VL-72B hosted on modal as the orchestrer, it as access to 3 MCP servers:

- [agora-mcp](https://github.com/Fewsats/agora-mcp)
- [fewsats-mcp](https://github.com/Fewsats/fewsats-mcp)
- [Vitrual Try-On MCP](./mcp_server.py) built as part of this project, can be found in the `mcp_server.py` file.

The virtual try on MCP uses [Flux-Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) combined with automatic masking, the model is hosted on modal.

Better docs comming soon.
