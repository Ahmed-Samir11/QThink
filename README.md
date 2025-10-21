---
title: QThink Agentic POC
emoji: ⚛️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# QThink Agentic Proof-of-Concept with Gradio UI

This repository contains the proof-of-concept (POC) for the QThink hackathon project, complete with an interactive Gradio user interface. This service is intended to be submitted as our "Inference Endpoint" link.

The interactive demo showcases the two core pillars of our project through a user-friendly UI:

1.  **Agentic Workflow Demo Tab**: This simulates our full agentic MCP (Multi-Context Protocol) workflow. It takes a user prompt and displays a complete "trace" of the conversation between our (mocked) Planner, Reasoner, and Verifier agents as they collaborate to solve the problem.
2.  **QuYAML Parser Tool Tab**: This allows users to test our novel, token-efficient QuYAML format. It accepts a QuYAML string and returns the generated QASM 2.0 code and a text-based circuit diagram.

This POC proves our architecture is sound and ready for the real K2 Think API in Stage 2.
