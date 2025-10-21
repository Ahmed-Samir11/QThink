from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time

from mock_k2_api import call_mock_planner, call_mock_reasoner, call_mock_verifier
from quyaml_parser import parse_quyaml_to_qiskit
from qiskit.qasm2 import dumps as to_qasm2_str

app = FastAPI(
    title="QThink POC Demo - Agentic Workflow",
    description="Proof-of-concept for the QThink Hackathon Project. This API demonstrates the full agentic MCP workflow.",
    version="0.2.0"
)

# --- Pydantic Models ---
class SimulationPrompt(BaseModel):
    prompt: str

class AgentStep(BaseModel):
    agent: str
    thought: str
    output: Any = Field(..., description="The output of the agent, can be a string, list, or dict.")

# --- Endpoints ---
@app.post("/tools/parse-quyaml-to-qasm")
async def parse_quyaml_endpoint(quyaml_body: str = Body(..., media_type="application/yaml",
                                                      example="# QYAML v0.1: Bell State...")):
    """
    **TOOL ENDPOINT:** Proves the QuYAML parser (SDS) is realistic.
    An agent would call this tool to convert QuYAML to QASM.
    """
    try:
        quantum_circuit = parse_quyaml_to_qiskit(quyaml_body)
        qasm_output = to_qasm2_str(quantum_circuit)
        return {
            "status": "success",
            "qasm_2_0_output": qasm_output,
            "text_diagram": str(quantum_circuit)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing QuYAML: {str(e)}")

@app.post("/solve/agentic-trace", response_model=Dict[str, Any])
async def agentic_trace_endpoint(body: SimulationPrompt):
    """
    **MAIN ENDPOINT: Proves the Agentic MCP (SRS) is sound.**
    
    This endpoint simulates the full Orchestrator workflow, showing the
    step-by-step 'trace' of agents (Planner, Reasoner, Verifier)
    collaborating to solve the user's prompt.
    """
    start_time = time.time()
    agent_trace: List[AgentStep] = []

    try:
        # --- Orchestrator: Step 1 - Call Planner Agent ---
        agent_trace.append(AgentStep(agent="Orchestrator", thought="Prompt received. Sending to Planner Agent.", output=f"Prompt: {body.prompt}"))
        plan = call_mock_planner(body.prompt)
        agent_trace.append(AgentStep(agent="Planner", thought="I have analyzed the prompt and created a step-by-step plan.", output=plan))

        # --- Orchestrator: Step 2 - Call Reasoner Agent in a loop ---
        reasoner_results = []
        for task in plan:
            agent_trace.append(AgentStep(agent="Orchestrator", thought=f"Dispatching task to Reasoner Agent: {task}", output=""))
            time.sleep(0.1) # Simulate network latency
            result = call_mock_reasoner(task)
            reasoner_results.append({"task": task, "result": result})
            agent_trace.append(AgentStep(agent="Reasoner", thought=f"I have completed task: {task}", output=result))

        # --- Orchestrator: Step 3 - Call Verifier Agent ---
        agent_trace.append(AgentStep(agent="Orchestrator", thought="All reasoning steps complete. Sending full results to Verifier Agent.", output=reasoner_results))
        verification = call_mock_verifier(agent_trace)
        agent_trace.append(AgentStep(agent="Verifier", thought="I have analyzed the full trace for correctness.", output=verification))
        
        end_time = time.time()
        
        return {
            "status": "success",
            "total_time_seconds": round(end_time - start_time, 2),
            "agent_trace": [step.dict() for step in agent_trace]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the QThink Agentic POC API. See the /docs for interactive endpoints."}
