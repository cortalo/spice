import re
from dataclasses import dataclass
from typing import List, Optional, Union, Dict
import numpy
import math

exp = math.exp
vt = 26e-3
isat = 1e-3

def eng_value(value_str: str) -> float:
    """
    Parse a string representation of a value and return the corresponding float.
    
    Args:
    value_str (str): A string representing a numeric value, potentially with suffixes like K, M, or scientific notation.
    
    Returns:
    float: The parsed numeric value.
    """
    value_str = value_str.upper()
    
    # Try to directly convert to float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle suffixes
    suffixes = {
        'T': 1e12,
        'G': 1e9,
        'MEG': 1e6,
        'K': 1e3,
        'M': 1e-3,
        'U': 1e-6,
        'N': 1e-9,
        'P': 1e-12,
        'F': 1e-15,
    }
    
    for suffix, multiplier in suffixes.items():
        if value_str.endswith(suffix):
            try:
                return float(value_str[:-1]) * multiplier
            except ValueError:
                pass
    
    # Handle scientific notation
    try:
        return float(value_str.replace('E', 'e'))
    except ValueError:
        pass
    
    # If all parsing attempts fail, raise an exception
    raise ValueError(f"Unable to parse value: {value_str}")


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


def get_nonlinear_indices(elements):
    nonlinear_indices = []
    for index, element in enumerate(elements):
        if element.element_type in 'DQM':
            nonlinear_indices.append(index)
    return nonlinear_indices



def v_stamp(matrix, rhs, element, positive=True):
    assert element.element_type == 'V'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    ie = element.num_nodes - 1 + element.g2_index
    if np >= 0: matrix[np, ie] += +1 if positive else -1
    if nm >= 0: matrix[nm, ie] += -1 if positive else +1
    if np >= 0: matrix[ie, np] += +1 if positive else -1
    if nm >= 0: matrix[ie, nm] += -1 if positive else +1
    rhs[ie, 0] += +val if positive else -val

def i_stamp(matrix, rhs, element, positive=True):
    assert element.element_type == 'I'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    if element.g2_index == None:
        if np >= 0: rhs[np, 0] += -val if positive else +val
        if nm >= 0: rhs[nm, 0] += +val if positive else -val
    else:
        ie = element.num_nodes - 1 + element.g2_index
        if np >= 0: matrix[np, ie] += +1 if positive else -1
        if nm >= 0: matrix[nm, ie] += -1 if positive else +1
        matrix[ie, ie] += -1 if positive else +1
        rhs[ie, 0] += +val if positive else -val

def r_stamp(matrix, rhs, element, positive=True):
    assert element.element_type == 'R'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    val = element.value
    if element.g2_index == None:
        if np >= 0: matrix[np, np] += +1/val if positive else -1/val
        if (np >= 0) and (nm >= 0): matrix[np, nm] += -1/val if positive else +1/val
        if (np >= 0) and (nm >= 0): matrix[nm, np] += -1/val if positive else +1/val
        if nm >= 0: matrix[nm, nm] += +1/val if positive else -1/val
    else:
        ie = element.num_nodes - 1 + element.g2_index
        if np >= 0: matrix[np, ie] += +1 if positive else -1
        if nm >= 0: matrix[nm, ie] += -1 if positive else +1
        if np >= 0: matrix[ie, np] += +1 if positive else -1
        if nm >= 0: matrix[ie, nm] += -1 if positive else +1
        matrix[ie, ie] += -val if positive else +val


f_op_g = lambda v: (isat/vt)*exp(v/vt)
f_op_r = lambda v: 1/f_op_g(v)
f_op_i = lambda v: isat*(exp(v/vt)-1)-(isat/vt)*v*exp(v/vt)
def d_stamp(matrix, rhs, element, positive=True, sol=None):
    assert element.element_type == 'D'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    if sol==None:
        sol = [0 for _ in range(len(rhs))]
    else:
        assert isinstance(sol, list)
    op_r = CircuitElement(
        element_type='R',
        number=0,
        node_plus=np+1,
        node_minus=nm+1,
        value=f_op_r((sol[np] if np >= 0 else 0) - (sol[nm] if nm >= 0 else 0))
    )
    r_stamp(matrix, rhs, op_r, positive)
    op_i = CircuitElement(
        element_type='I',
        number=0,
        node_plus=np+1,
        node_minus=nm+1,
        value=f_op_i((sol[np] if np >= 0 else 0) - (sol[nm] if nm >= 0 else 0))
    )
    i_stamp(matrix, rhs, op_i, positive)


def stamp(matrix, rhs, element, positive=True, sol=None):
    if element.element_type == 'V':
        v_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'I':
        i_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'R':
        r_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'D':
        d_stamp(matrix, rhs, element, positive, sol)
    else:
        raise ValueError(f"Unknown element type for stamp: {element.element_type}")



def init_matrix(elements, matrix, rhs):
    for element in elements:
        stamp(matrix, rhs, element)

def dc(elements):
    all_nodes = get_all_nodes(elements)
    group2_indices = get_group2_indices(elements, len(all_nodes))
    nonlinear_indices = get_nonlinear_indices(elements)
    matrix_dim = len(all_nodes)-1+len(group2_indices)
    matrix = numpy.zeros((matrix_dim,matrix_dim))
    rhs = numpy.zeros((matrix_dim,1))
    init_matrix(elements, matrix, rhs)
    if nonlinear_indices == []:
        print(numpy.linalg.solve(matrix, rhs))
    else:
        prev_sol = [0 for _ in range(matrix_dim)]
        cur_sol = [float(item[0]) for item in numpy.linalg.solve(matrix, rhs)]
        for i in range(10):
            for j in nonlinear_indices:
                stamp(matrix, rhs, elements[j], positive=False, sol=prev_sol)
                stamp(matrix, rhs, elements[j], positive=True, sol=cur_sol)
            prev_sol = cur_sol
            cur_sol = [float(item[0]) for item in numpy.linalg.solve(matrix, rhs)]
            print(prev_sol)


@dataclass
class CircuitElement:
    element_type: str
    number: str
    node_plus: int
    node_minus: int
    value: Union[float, str]
    group: Optional[int] = None
    node_c: Optional[int] = None
    node_b: Optional[int] = None
    node_e: Optional[int] = None
    node_d: Optional[int] = None
    node_g: Optional[int] = None
    node_s: Optional[int] = None
    g2_index: Optional[int] = None
    num_nodes: Optional[int] = None

@dataclass
class Model:
    name: str
    model_type: str
    parameters: Dict[str, float]

    def __init__(self, name: str, model_type: str, parameters: Dict[str, str]):
        self.name = name
        self.model_type = model_type
        self.parameters = {key: eng_value(value) for key, value in parameters.items()}

    def get_parameter(self, param_name: str, default: float = None) -> float:
        return self.parameters.get(param_name, default)


# V must in group 2
class CircuitParser:
    def __init__(self):
        self.elements: List[CircuitElement] = []
        self.models: Dict[str, Dict[str, float]] = {}

    def parse_file(self, filename: str) -> None:
        with open(filename, 'r') as file:
            i = 0
            for line in file:
                if (i == 0):
                    i += 1
                    continue
                self.parse_line(line)

    def parse_line(self, line: str) -> None:
        # Remove comments and leading/trailing whitespace
        line = re.sub(r'%.*$', '', line).strip()
        if not line:
            return

        # Split the line into tokens
        tokens = re.split(r'\s+', line)
        element_type = tokens[0][0].upper()

        if element_type in '*':
            pass
        elif tokens[0].upper() == '.MODEL':
            self.parse_model(tokens)
        elif element_type in 'VI':
            self.parse_vi(tokens)
        elif element_type == 'R':
            self.parse_r(tokens)
        elif element_type in 'CL':
            self.parse_cl(tokens)
        elif element_type == 'D':
            self.parse_d(tokens)
        elif element_type in 'QM':
            self.parse_qm(tokens)
        elif element_type in '.':
            pass
        else:
            raise ValueError(f"Unknown element type: {element_type}")

    def parse_model(self, tokens: List[str]) -> None:
        # Extract model name and type
        model_name = tokens[1]
        model_type = tokens[2]

        # Initialize parameters dictionary
        parameters = {}

        # Parse parameters
        for param in tokens[3:]:
            key, value = param.split('=')
            parameters[key.strip()] = value.strip()

        # Create Model instance
        model = Model(model_name, model_type, parameters)

        # Add model to the models dictionary
        self.models[model_name] = model


    def parse_vi(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][0],
            number=str(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if tokens[0][0] == 'V':
            element.group = 2
        self.elements.append(element)

    def parse_r(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type='R',
            number=str(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if len(tokens) > 4 and tokens[4].upper() == 'G2':
            element.group = 2
        self.elements.append(element)

    def parse_cl(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][0],
            number=str(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if len(tokens) > 4:
            element.group = 2
        self.elements.append(element)

    def parse_d(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type='D',
            number=str(tokens[0][1:]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=1.0  # Default value
        )
        if len(tokens) > 3:
            element.value = float(tokens[3])
        self.elements.append(element)

    def parse_qm(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][:1],
            number=str(tokens[0][1:]),
            node_c=int(tokens[1]),
            node_b=int(tokens[2]),
            node_e=int(tokens[3]),
            value=str(tokens[4]),
            node_plus=int(tokens[1]),  # Set node_plus to collector
            node_minus=int(tokens[3])  # Set node_minus to emitter
        )
        self.elements.append(element)

    def get_elements(self) -> List[CircuitElement]:
        return self.elements


    def get_models(self) -> Dict[str, str]:
        return self.models

