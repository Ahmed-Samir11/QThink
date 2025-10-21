---
title: QThink Agentic POC
emoji: ⚛️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# QThink Agentic Proof-of-Concept API

This repository contains the proof-of-concept (POC) API for the QThink hackathon project. This service is intended to be submitted as our "Inference Endpoint" link.

It demonstrates the two core pillars of our project:

1.  **`/tools/parse-quyaml-to-qasm`**: A tool endpoint that our agent uses. It accepts our novel, token-efficient QuYAML format and returns valid QASM 2.0. This proves our custom parser works.
2.  **`/solve/agentic-trace`**: The **main demo endpoint**. This simulates our full agentic MCP (Multi-Context Protocol) workflow. It takes a user prompt and returns a complete "trace" of the conversation between our (mocked) Planner, Reasoner, and Verifier agents as they collaborate to solve the problem.

This POC proves our architecture is sound and ready for the real K2 Think API in Stage 2. See the `/docs` endpoint for a full FastAPI interactive demo.
