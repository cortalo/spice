#!/usr/bin/env python3

from parser import *


# Usage example
parser = CircuitParser()
parser.parse_file('bjt.ckt')
elements = parser.get_elements()
models = parser.get_models()
# Print parsed elements
print(elements)
print(models)
dc(elements)
