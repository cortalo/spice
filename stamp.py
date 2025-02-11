import math
from parser import CircuitElement

exp = math.exp
vt = 26e-3
isat = 1e-3



def get_bjt_model(model):
    ret = {}
    ret['IS'] = model.get_parameter('IS', 1e-16)
    ret['BF'] = model.get_parameter('BF', 100)
    return ret

def get_all_nodes(elements):
    nodes = set()
    for index, element in enumerate(elements.values()):
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
    for index, element in enumerate(elements.values()):
        element.num_nodes = num_nodes
        if hasattr(element, 'group') and element.group == 2:
            element.g2_index = element_g2_index
            element_g2_index += 1
            group2_indices.append(index)
        elif element.number_c is not None:
            ctrl_element = elements[element.number_c]
            if ctrl_element.group != 2:
                ctrl_element.group = 2
                ctrl_element.g2_index = element_g2_index
                element_g2_index += 1
                group2_indices.append(index)
            element.g2_index_c = ctrl_element.g2_index
    return group2_indices


def get_nonlinear_indices(elements):
    nonlinear_indices = []
    for index, element in enumerate(elements.values()):
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

def g_stamp(matrix, rhs, element, positive=True):
    assert element.element_type == 'G'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    ncp = element.node_cpuls - 1
    ncm = element.node_cminus - 1
    val = element.value
    if (np >= 0) and (ncp >= 0): matrix[np, ncp] += +val if positive else -val
    if (np >= 0) and (ncm >= 0): matrix[np, ncm] += -val if positive else +val
    if (nm >= 0) and (ncp >= 0): matrix[nm, ncp] += -val if positive else +val
    if (nm >= 0) and (ncm >= 0): matrix[nm, ncm] += +val if positive else -val

def f_stamp(matrix, rhs, element, positive=True):
    assert element.element_type == 'F'
    np = element.node_plus - 1
    nm = element.node_minus - 1
    iec = element.num_nodes - 1 + element.g2_index_c
    val = element.value
    if np >= 0: matrix[np, iec] += +val if positive else -val
    if nm >= 0: matrix[nm, iec] += -val if positive else +val


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

def bjt_alpha_f(model):
    return model['BF']/(model['BF']+1)
def q_stamp(matrix, rhs, element, positive=True, sol=None, models=None):
    assert element.element_type == 'Q'
    nb = element.node_b - 1
    nc = element.node_c - 1
    ne = element.node_e - 1
    if element.model is None:
        element.model = get_bjt_model(models[element.value])
        print(element.model)
    if sol==None:
        sol = [0 for _ in range(len(rhs))]
    else:
        assert isinstance(sol, list)

def stamp(matrix, rhs, element, positive=True, sol=None, models=None):
    if element.element_type == 'V':
        v_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'I':
        i_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'R':
        r_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'G':
        g_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'F':
        f_stamp(matrix, rhs, element, positive)
    elif element.element_type == 'D':
        d_stamp(matrix, rhs, element, positive, sol)
    elif element.element_type == 'Q':
        q_stamp(matrix, rhs, element, positive, sol, models)
    else:
        raise ValueError(f"Unknown element type for stamp: {element.element_type}")
