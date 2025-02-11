#!/usr/bin/env python3

from parser import CircuitParser
from stamp import *
from analysis import *


# Usage example
parser = CircuitParser()
parser.parse_file('cccs.ckt')
elements = parser.get_elements()
models = parser.get_models()
# Print parsed elements
dc(elements, models)
