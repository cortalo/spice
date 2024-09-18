import re
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

import numpy as np

def print_matrix_vector(M, V):
    """
    Print a matrix M and a vector V side by side.

    :param M: NxN NumPy array (square matrix)
    :param V: Nx1 NumPy array (column vector)
    """
    # Ensure M is square and V has the same number of rows as M
    assert M.shape[0] == M.shape[1], "M must be a square matrix"
    assert M.shape[0] == V.shape[0], "V must have the same number of rows as M"
    assert V.shape[1] == 1, "V must be a column vector"

    # Convert V to a 2D array for consistent formatting
    V_2D = V.reshape(-1, 1)

    # Combine M and V horizontally
    combined = np.hstack((M, V_2D))

    # Determine the width for formatting based on the largest absolute value
    max_val = max(np.abs(combined).max(), 1)
    width = max(8, int(np.log10(max_val)) + 6)

    # Print the header
    print(f"{'M'.ljust(width * M.shape[1])}V")
    print("-" * (width * (M.shape[1] + 1)))

    # Custom format function
    def format_number(x):
        if abs(x) < 1e-10:  # Consider values very close to 0 as 0
            return f"{'0':>{width}}"
        else:
            return f"{x:{width}.4f}"

    for row in combined:
        # Format each number in the row
        formatted_row = [format_number(x) for x in row]

        # Join the M part and the V part
        print("".join(formatted_row[:-1]) + " | " + formatted_row[-1])


def get_all_nodes(elements):
    nodes = set()
    for element in elements:
        # Add node_plus and node_minus for all elements
        nodes.add(element.node_plus)
        nodes.add(element.node_minus)

        # Add additional nodes for transistors
        if element.element_type.startswith('Q') or element.element_type.startswith('M'):
            nodes.add(element.node_c)
            nodes.add(element.node_b)
            nodes.add(element.node_e)

    # The total number of nodes is the length of the set
    return list(nodes)

def get_group2_indices(elements, num_nodes):
    group2_indices = []
    element_g2_index = 0
    for index, element in enumerate(elements):
        element.num_nodes = num_nodes
        if hasattr(element, 'group') and element.group == 2:
            element.g2_index = element_g2_index
            element_g2_index += 1
            group2_indices.append(index)
    return group2_indices


def v_stamp(matrix, rhs, element):
    assert element.element_type == 'V'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    ie = element.num_nodes - 1 + element.g2_index
    if np >= 0: matrix[np, ie] += +1
    if nm >= 0: matrix[nm, ie] += -1
    if np >= 0: matrix[ie, np] += +1
    if nm >= 0: matrix[ie, nm] += -1
    rhs[ie, 0] += +val

def i_stamp(matrix, rhs, element):
    assert element.element_type == 'I'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    if element.g2_index == None:
        if np >= 0: rhs[np, 0] += -val
        if nm >= 0: rhs[nm, 0] += +val
    else:
        ie = element.num_nodes - 1 + element.g2_index
        if np >= 0: matrix[np, ie] += +1
        if nm >= 0: matrix[nm, ie] += -1
        matrix[ie, ie] += -1
        rhs[ie, 0] += +val

def r_stamp(matrix, rhs, element):
    assert element.element_type == 'R'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    if element.g2_index == None:
        if np >= 0: matrix[np, np] += +1/val
        if (np >= 0) and (nm >= 0): matrix[np, nm] += -1/val
        if (np >= 0) and (nm >= 0): matrix[nm, np] += -1/val
        if nm >= 0: matrix[nm, nm] += +1/val
    else:
        ie = element.num_nodes - 1 + element.g2_index
        if np >= 0: matrix[np, ie] += +1
        if nm >= 0: matrix[nm, ie] += -1
        if np >= 0: matrix[ie, np] += +1
        if nm >= 0: matrix[ie, nm] += -1
        matrix[ie, ie] += -val

def stamp(matrix, rhs, element):
    if element.element_type == 'V':
        v_stamp(matrix, rhs, element)
    elif element.element_type == 'I':
        i_stamp(matrix, rhs, element)
    elif element.element_type == 'R':
        r_stamp(matrix, rhs, element)
    else:
        raise ValueError(f"Unknown element type for stamp: {element.element_type}")



def get_matrix(elements):
    all_nodes = get_all_nodes(elements)
    group2_indices = get_group2_indices(elements, len(all_nodes))
    matrix_dim = len(all_nodes)-1+len(group2_indices)
    matrix = np.zeros((matrix_dim,matrix_dim))
    rhs = np.zeros((matrix_dim,1))
    for element in elements:
        stamp(matrix, rhs, element)
    print(np.linalg.solve(matrix, rhs))

@dataclass
class CircuitElement:
    element_type: str
    number: int
    node_plus: int
    node_minus: int
    value: float
    group: Optional[int] = None
    node_c: Optional[int] = None
    node_b: Optional[int] = None
    node_e: Optional[int] = None
    node_d: Optional[int] = None
    node_g: Optional[int] = None
    node_s: Optional[int] = None
    g2_index: Optional[int] = None
    num_nodes: Optional[int] = None

# V must in group 2
class CircuitParser:
    def __init__(self):
        self.elements: List[CircuitElement] = []

    def parse_file(self, filename: str) -> None:
        with open(filename, 'r') as file:
            for line in file:
                self.parse_line(line)

    def parse_line(self, line: str) -> None:
        # Remove comments and leading/trailing whitespace
        line = re.sub(r'%.*$', '', line).strip()
        if not line:
            return

        # Split the line into tokens
        tokens = re.split(r'\s+', line)
        element_type = tokens[0][0].upper()

        if element_type in 'VI':
            self.parse_vi(tokens)
        elif element_type == 'R':
            self.parse_r(tokens)
        elif element_type in 'CL':
            self.parse_cl(tokens)
        elif element_type == 'D':
            self.parse_d(tokens)
        elif element_type in 'QM':
            self.parse_qm(tokens)
        else:
            raise ValueError(f"Unknown element type: {element_type}")

    def parse_vi(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][0],
            number=int(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=float(tokens[3])
        )
        if tokens[0][0] == 'V':
            element.group = 2
        self.elements.append(element)

    def parse_r(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type='R',
            number=int(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=float(tokens[3])
        )
        if len(tokens) > 4 and tokens[4].upper() == 'G2':
            element.group = 2
        self.elements.append(element)

    def parse_cl(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][0],
            number=int(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=float(tokens[3])
        )
        if len(tokens) > 4:
            element.group = 2
        self.elements.append(element)

    def parse_d(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type='D',
            number=int(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=1.0  # Default value
        )
        if len(tokens) > 3:
            element.value = float(tokens[3])
        self.elements.append(element)

    def parse_qm(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][:2],
            number=int(tokens[0][2:]),
            node_c=int(tokens[1]),
            node_b=int(tokens[2]),
            node_e=int(tokens[3]),
            value=1.0,  # Default value
            node_plus=int(tokens[1]),  # Set node_plus to collector
            node_minus=int(tokens[3])  # Set node_minus to emitter
        )
        if len(tokens) > 4:
            element.value = float(tokens[4])
        self.elements.append(element)

    def get_elements(self) -> List[CircuitElement]:
        return self.elements

# Usage example
parser = CircuitParser()
parser.parse_file('circuit_specification.txt')
elements = parser.get_elements()

# Print parsed elements
print(get_matrix(elements))
