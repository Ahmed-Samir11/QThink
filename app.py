import gradio as gr
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time

# --- Import All Project Logic ---
from mock_k2_api import call_mock_planner, call_mock_reasoner, call_mock_verifier
from quyaml_parser import parse_quyaml_to_qiskit
from qiskit.qasm2 import dumps as to_qasm2_str

# === 1. BACKEND LOGIC (FastAPI) ===
# This is our existing FastAPI app. Gradio will wrap around it.
app = FastAPI(
    title="QThink Agentic POC API",
    description="This API powers the QThink Gradio Demo.",
    version="0.3.0"
)

# --- Pydantic Models ---
class SimulationPrompt(BaseModel):
    prompt: str

class AgentStep(BaseModel):
    agent: str
    thought: str
    output: Any

# --- API Endpoints ---
@app.post("/tools/parse-quyaml-to-qasm")
async def parse_quyaml_endpoint(quyaml_body: str = Body(...)):
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

@app.post("/solve/agentic-trace")
async def agentic_trace_endpoint(body: SimulationPrompt):
    try:
        agent_trace: List[AgentStep] = []
        plan = call_mock_planner(body.prompt)
        agent_trace.append(AgentStep(agent="Planner", thought="I have analyzed the prompt and created a step-by-step plan.", output=plan))
        
        reasoner_results = []
        for task in plan:
            result = call_mock_reasoner(task)
            reasoner_results.append({"task": task, "result": result})
            agent_trace.append(AgentStep(agent="Reasoner", thought=f"I have completed task: {task}", output=result))

        verification = call_mock_verifier(agent_trace)
        agent_trace.append(AgentStep(agent="Verifier", thought="I have analyzed the full trace for correctness.", output=verification))
        
        return {"agent_trace": [step.dict() for step in agent_trace]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === 2. FRONTEND LOGIC (Gradio UI) ===
# These functions will call our backend logic.

# --- Function for the QuYAML Parser Tab ---
def parse_quyaml_interface(quyaml_string):
    """Gradio interface function for the QuYAML parser."""
    try:
        circuit = parse_quyaml_to_qiskit(quyaml_string)
        qasm = to_qasm2_str(circuit)
        diagram = str(circuit)
        return qasm, diagram
    except Exception as e:
        return f"Error: {str(e)}", ""

# --- Function for the Agentic Trace Tab ---
def agent_trace_interface(prompt):
    """Gradio interface function for the agentic trace."""
    if not prompt:
        return "Please enter a prompt.", ""
        
    trace_log = "Simulating agentic workflow...\n\n"
    
    plan = call_mock_planner(prompt)
    trace_log += "--- Planner Agent ---\n"
    trace_log += "Plan:\n" + "\n".join([f"- {p}" for p in plan]) + "\n\n"
    
    trace_log += "--- Reasoner Agent ---\n"
    all_results = []
    for task in plan:
        result = call_mock_reasoner(task)
        all_results.append(result)
        trace_log += f"Executing Task: {task}\nResult: {result}\n"
    
    trace_log += "\n--- Verifier Agent ---\n"
    verification = call_mock_verifier(all_results)
    trace_log += f"Verification Status: {verification['status']}\nDetails: {verification}\n"
    
    # We will return the final verification dictionary and the full text log
    return verification, trace_log

# --- Define the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="QThink Demo") as demo:
    gr.Markdown("# ⚛️ QThink: Agentic Quantum Co-Pilot")
    gr.Markdown("This interface demonstrates the two core features of our hackathon project. Use the tabs below to explore.")

    with gr.Tabs():
        with gr.TabItem("Agentic Workflow Demo"):
            gr.Markdown("### Simulate the full agentic workflow.")
            with gr.Row():
                prompt_input = gr.Textbox(lines=5, label="User Prompt", placeholder="Enter your prompt here...")
            execute_button = gr.Button("Execute Workflow", variant="primary")
            gr.Markdown("### Agent Trace Log")
            trace_output_log = gr.Textbox(lines=20, label="Full Agent Conversation Log", interactive=False)
            gr.Markdown("### Final Verification Output")
            trace_output_final = gr.JSON(label="Final Output from Verifier Agent")

            example_prompt = gr.Examples(
                examples=["Simulate a 3-qubit Quantum Fourier Transform on the state |101⟩."],
                inputs=prompt_input
            )

        with gr.TabItem("QuYAML Parser Tool"):
            gr.Markdown("### Test our novel, token-efficient QuYAML format.")
            with gr.Row():
                quyaml_input = gr.Textbox(lines=15, label="QuYAML Input", placeholder="Enter your QuYAML circuit definition here...")
                with gr.Column():
                    qasm_output = gr.Code(label="Generated QASM 2.0", language="qasm")
                    diagram_output = gr.Textbox(label="Text Circuit Diagram", interactive=False)
            
            parse_button = gr.Button("Parse QuYAML", variant="primary")
            
            example_quyaml = gr.Examples(
                examples=[
                    ["""# QYAML v0.1: Bell State
circuit: BellState
qreg: q[2]
creg: c[2]
instructions:
  - h q[0]
  - cx q[0], q[1]
  - measure q, c"""]
                ],
                inputs=quyaml_input
            )

    # --- Connect functions to interfaces ---
    parse_button.click(fn=parse_quyaml_interface, inputs=quyaml_input, outputs=[qasm_output, diagram_output])
    execute_button.click(fn=agent_trace_interface, inputs=prompt_input, outputs=[trace_output_final, trace_output_log])

# === 3. MOUNTING & LAUNCHING ===
# Mount the Gradio UI onto the FastAPI backend
app = gr.mount_gradio_app(app, demo, path="/")

# If you want to run this file locally, you can add this:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)
