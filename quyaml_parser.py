import yaml
from qiskit import QuantumCircuit
import re
import numpy as np

def parse_quyaml_to_qiskit(quyaml_string: str) -> QuantumCircuit:
    """
    Parses a QuYAML string into a Qiskit QuantumCircuit object.
    """
    try:
        data = yaml.safe_load(quyaml_string)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid QuYAML format: {e}")

    circuit_name = data.get('circuit', 'my_circuit')
    
    def get_reg_size(reg_str):
        match = re.search(r'\[(\d+)\]', reg_str)
        return int(match.group(1)) if match else 0
        
    q_size = get_reg_size(data.get('qreg', 'q[0]'))
    c_size = get_reg_size(data.get('creg', 'c[0]'))
    
    if q_size == 0:
        raise ValueError("QuYAML must define at least one quantum register (e.g., qreg: q[1])")
        
    qc = QuantumCircuit(q_size, c_size, name=circuit_name)

    instructions = data.get('instructions', [])
    for inst_str in instructions:
        apply_instruction(qc, inst_str)
        
    return qc

def apply_instruction(qc: QuantumCircuit, inst_str: str):
    """
    Parses a single QuYAML instruction string and applies it to the circuit.
    """
    parts = inst_str.split()
    gate = parts[0].lower()
    
    def get_indices(target_strings):
        indices = []
        for s in target_strings:
            match = re.search(r'\[(\d+)\]', s)
            if match:
                indices.append(int(match.group(1)))
        return indices

    targets = [p.replace(',', '') for p in parts[1:]]
    q_indices = get_indices(targets)

    try:
        if gate == 'h':
            qc.h(q_indices[0])
        elif gate == 'x':
            qc.x(q_indices[0])
        elif gate == 'cx':
            qc.cx(q_indices[0], q_indices[1])
        elif gate == 'swap':
            qc.swap(q_indices[0], q_indices[1])
        elif gate.startswith('cphase'):
            angle_str = re.search(r'\((.*?)\)', gate).group(1)
            angle_map = {'pi/2': np.pi / 2, 'pi/4': np.pi / 4, 'pi': np.pi}
            angle = angle_map.get(angle_str, float(angle_str))
            qc.cp(angle, q_indices[0], q_indices[1])
        elif gate == 'measure':
            qc.measure(range(qc.num_qubits), range(qc.num_clbits))
    except Exception as e:
        raise ValueError(f"Could not parse instruction '{inst_str}'. Error: {e}")
