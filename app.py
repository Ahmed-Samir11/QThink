import gradio as gr
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import numpy as np
import random

# --- Import All Project Logic ---
from mock_k2_api import call_mock_planner, call_mock_reasoner, call_mock_verifier
from quyaml_parser import parse_quyaml_to_qiskit
from qiskit.qasm2 import dumps as to_qasm2_str

# Ground truth for automated evaluation (QFT on |101⟩)
GROUND_TRUTH_QFT_101_VECTOR = np.array([
    0.35355339+0.j, -0.35355339+0.j, 0.0+0.35355339j, 0.0-0.35355339j,
   -0.25-0.25j,      0.25+0.25j,     0.25-0.25j,     -0.25+0.25j
])

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

# === DGM HELPER FUNCTIONS ===

def call_mock_dgm_self_improve(current_prompt: str, failure_analysis: str) -> str:
    """Simulates the DGM's self-modification by suggesting a prompt improvement."""
    suggestions = [
        "# DGM Suggestion: Add a directive to show mathematical derivations for each step.",
        "# DGM Suggestion: Rephrase to request complex numbers in 'a + bi' format explicitly.",
        "# DGM Suggestion: Instruct the model to double-check phase calculations before proceeding."
    ]
    suggestion = random.choice(suggestions)
    return f"{current_prompt}\n\n{suggestion}\n# Justification: {failure_analysis}"

def extract_final_vector_from_trace(agent_trace: list) -> Optional[np.ndarray]:
    """Parses the final vector string from a mock agent trace."""
    try:
        # In our mock, the verifier's output contains the final vector preview
        verifier_step = next(step for step in reversed(agent_trace) if step['agent'] == 'Verifier')
        vector_str = verifier_step['output']['final_state_preview']
        # This is a simplified parser for the mock format: "[0.353, -0.353, 0.353i, ...]"
        vector_str = vector_str.strip('[]').replace('i', 'j')
        parts = [complex(p.strip()) for p in vector_str.split(',')]
        return np.array(parts)
    except (StopIteration, ValueError, TypeError):
        # Fallback for a slightly different vector format (to simulate prompt-induced changes)
        try:
            verifier_step = next(step for step in reversed(agent_trace) if step['agent'] == 'Verifier')
            vector_str = verifier_step['output']['final_state_preview']
            vector_str = vector_str.strip('[]').replace('i', 'j')
            # Handle format like "-0.353(1+j)"
            parts = [eval(p.strip().replace('(', '*(')) for p in vector_str.split(',') if p.strip()]
            return np.array(parts)
        except:
             return None # Failed to parse

def run_automated_evaluation(new_prompt_set: dict) -> dict:
    """Runs the mock simulation and evaluates the result against ground truth."""
    # Simulate the agentic trace using the new prompt logic (mocked)
    prompt_for_test_case = "Simulate a 3-qubit Quantum Fourier Transform on the state |101⟩."
    plan = call_mock_planner(prompt_for_test_case)
    trace = []
    for task in plan:
        result = call_mock_reasoner(task)
        trace.append({'agent': 'Reasoner', 'output': result})
    verification = call_mock_verifier(trace)
    trace.append({'agent': 'Verifier', 'output': verification})

    # Extract the final vector from this simulated trace
    predicted_vector = extract_final_vector_from_trace(trace)

    if predicted_vector is None:
        return {'status': 'Error', 'message': 'Evaluation failed: Could not parse final vector from agent output.'}

    # Normalize vectors for fair comparison
    predicted_vector /= np.linalg.norm(predicted_vector)
    
    # Calculate Mean Squared Error
    mse = np.mean(np.abs(GROUND_TRUTH_QFT_101_VECTOR - predicted_vector)**2)
    threshold = 0.01
    result = "PASSED" if mse < threshold else "FAILED"

    return {
        'status': 'Completed',
        'test_case': 'QFT(|101⟩)',
        'mse': f"{mse:.6f}",
        'threshold': threshold,
        'result': result
    }

# --- Function for the DGM Tab ---
def dgm_interface(current_prompt, failure_analysis):
    """Gradio interface function for the DGM cycle."""
    if not failure_analysis:
        return "Please provide a failure analysis to guide the improvement."
        
    # Step 1: Self-Modification (Suggest a new prompt)
    suggested_prompt = call_mock_dgm_self_improve(current_prompt, failure_analysis)
    
    # Step 2: Automated Evaluation (Test the new prompt)
    new_prompt_set = {'reasoner': suggested_prompt} # In a real app, this would be a full set
    evaluation_result = run_automated_evaluation(new_prompt_set)
    
    # Step 3: Format the output for the UI
    output_log = "--- DGM SELF-MODIFICATION ---\n"
    output_log += "Generated a new candidate prompt based on failure analysis:\n"
    output_log += f'"""{suggested_prompt}"""\n\n'
    output_log += "--- AUTOMATED EVALUATION ---\n"
    output_log += f"Running benchmark test: {evaluation_result.get('test_case', 'N/A')}\n"
    output_log += f"Mean Squared Error (MSE): {evaluation_result.get('mse', 'N/A')}\n"
    output_log += f"Success Threshold: MSE < {evaluation_result.get('threshold', 'N/A')}\n"
    output_log += "--------------------------------\n"
    output_log += f"EVALUATION RESULT: {evaluation_result.get('result', 'ERROR')}\n"
    output_log += "--------------------------------\n"
    if evaluation_result.get('result') == 'PASSED':
        output_log += "Conclusion: The suggested prompt is an improvement and will be added to the agent archive."
    else:
        output_log += "Conclusion: The suggested prompt did not pass the benchmark and will be discarded."
        
    return output_log

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
                    qasm_output = gr.Code(label="Generated QASM 2.0", language="python")
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

        with gr.TabItem("DGM Self-Improvement Cycle"):
            gr.Markdown("### Simulate the Darwin Gödel Machine Loop on Prompts")
            gr.Markdown("This tab demonstrates how QThink can self-improve. It takes a 'failure analysis' of a previous run, uses an LLM to suggest a new prompt, and then **automatically evaluates** if the new prompt produces a better result on a benchmark task.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    dgm_current_prompt = gr.Textbox(
                        value="You are a powerful mathematical and logical reasoning engine. Your task is to solve the following step precisely and show your work.",
                        lines=5, 
                        label="Current 'Reasoner' Prompt (Version 0)"
                    )
                    dgm_failure_input = gr.Textbox(
                        lines=3,
                        label="Failure Analysis / Improvement Goal",
                        placeholder="e.g., The mathematical explanation for complex number steps was unclear."
                    )
                    dgm_improve_button = gr.Button("Run DGM Improvement Cycle", variant="primary")

                with gr.Column(scale=2):
                    dgm_output_log = gr.Textbox(
                        lines=15, 
                        label="DGM Cycle Result (Suggested Prompt & Automated Evaluation)", 
                        interactive=False,
                        placeholder="Results of the self-improvement cycle will appear here..."
                    )

    # --- Connect functions to interfaces ---
    parse_button.click(fn=parse_quyaml_interface, inputs=quyaml_input, outputs=[qasm_output, diagram_output])
    execute_button.click(fn=agent_trace_interface, inputs=prompt_input, outputs=[trace_output_final, trace_output_log])
    dgm_improve_button.click(fn=dgm_interface, inputs=[dgm_current_prompt, dgm_failure_input], outputs=dgm_output_log)

# === 3. MOUNTING & LAUNCHING ===
# Mount the Gradio UI onto the FastAPI backend
app = gr.mount_gradio_app(app, demo, path="/")

# If you want to run this file locally, you can add this:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)
