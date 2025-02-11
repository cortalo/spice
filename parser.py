import re
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import numpy
import math


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
    node_cpuls: Optional[int] = None
    node_cminus: Optional[int] = None
    number_c: Optional[str] = None
    g2_index: Optional[int] = None
    g2_index_c: Optional[int] = None
    num_nodes: Optional[int] = None
    model: Optional[Dict[str, Any]] = None


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
        self.elements: Dict[str, CircuitElement] = {}
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
        elif element_type in 'GE':
            self.parse_ge(tokens)
        elif element_type in 'FH':
            self.parse_fh(tokens)
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
            number=str(tokens[0]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if tokens[0][0] == 'V':
            element.group = 2
        self.elements[tokens[0]] = element

    def parse_r(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type='R',
            number=str(tokens[0]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if len(tokens) > 4 and tokens[4].upper() == 'G2':
            element.group = 2
        self.elements[tokens[0]] = element

    def parse_cl(self, tokens: List[str]) -> None:
        element = CircuitElement(
            element_type=tokens[0][0],
            number=str(tokens[0]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            value=eng_value(tokens[3])
        )
        if len(tokens) > 4:
            element.group = 2
        self.elements[tokens[0]] = element
    
    def parse_ge(self, tokens: List[str]) -> None:
        '''
        VCCS:G
        VCVS:E
        '''
        element = CircuitElement(
            element_type=tokens[0][0],
            number=str(tokens[0]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            node_cpuls=int(tokens[3]),
            node_cminus=int(tokens[4]),
            value=eng_value(tokens[5])
        )
        self.elements[tokens[0]] = element

    def parse_fh(self, tokens: List[str]) -> None:
        '''
        CCCS:F
        CCVS:H
        '''
        element = CircuitElement(
            element_type=tokens[0][0],
            number=str(tokens[0]),
            node_plus=int(tokens[1]),
            node_minus=int(tokens[2]),
            number_c=str(tokens[3]),
            value=eng_value(tokens[4])
        )
        self.elements[tokens[0]] = element

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
        self.elements[tokens[0]] = element

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
        self.elements[tokens[0]] = element

    def get_elements(self) -> List[CircuitElement]:
        return self.elements


    def get_models(self) -> Dict[str, str]:
        return self.models

