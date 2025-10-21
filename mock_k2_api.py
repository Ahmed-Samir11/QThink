# This file simulates the K2 Think API for each agent in the MCP.

MOCK_QFT_PLAN = [
    "Task 1: Define initial 8D state vector for |101⟩.",
    "Task 2: Apply Hadamard on qubit 0.",
    "Task 3: Apply CPHASE(pi/2) from q[1] to q[0].",
    "Task 4: Apply CPHASE(pi/4) from q[2] to q[0].",
    "Task 5: Apply Hadamard on qubit 1.",
    "Task 6: Apply CPHASE(pi/2) from q[2] to q[1].",
    "Task 7: Apply Hadamard on qubit 2.",
    "Task 8: Apply SWAP(0, 2).",
    "Task 9: Consolidate and present final state vector."
]

MOCK_QFT_STEP_RESULTS = {
    "Task 1": "Initial state |101⟩ is index 5. Vector: [0, 0, 0, 0, 0, 1, 0, 0]",
    "Task 2": "Applying H(0). State becomes (1/sqrt(2)) * [0, 0, 0, 0, 1, -1, 0, 0]",
    "Task 3": "Applying CPHASE(pi/2). Control q[1] is 0 for all non-zero states. Vector is unchanged.",
    "Task 4": "Applying CPHASE(pi/4). Control q[2]=1, Target q[0]=1 for index 5. Phase e^(i*pi/4) is applied.",
    "Task 5": "Applying H(1). Amplitudes for indices 4,5,6,7 are calculated.",
    "Task 6": "Applying CPHASE(pi/2). Phase 'i' applied to indices 6 and 7.",
    "Task 7": "Applying H(2). Final amplitudes calculated across all 8 indices.",
    "Task 8": "Applying SWAP(0, 2). Indices [1,4], [3,6] are permuted.",
    "Task 9": "Final state vector consolidated. Ready for verification."
}

MOCK_QFT_VERIFICATION = {
    "status": "Verified",
    "steps_checked": 9,
    "mathematical_consistency": "High",
    "final_state_hash": "a3f4b01e2c5d6f7a... (mock hash)",
    "final_state_preview": "[0.353, -0.353, 0.353i, -0.353i, -0.25-0.25i, 0.25+0.25i, ...]"
}

def call_mock_planner(prompt: str) -> list:
    """Simulates the Planner Agent."""
    if "QFT" in prompt and "|101⟩" in prompt:
        return MOCK_QFT_PLAN
    return ["Task 1: Understand user prompt.", "Task 2: Formulate generic plan."]

def call_mock_reasoner(task: str) -> str:
    """Simulates the Reasoner Agent executing one task."""
    return MOCK_QFT_STEP_RESULTS.get(task, f"Mock execution result for: {task}")

def call_mock_verifier(full_trace: list) -> dict:
    """Simulates the Verifier Agent checking the whole process."""
    if len(full_trace) > 5:
        return MOCK_QFT_VERIFICATION
    return {"status": "Verification Failed", "reason": "Trace too short."}
