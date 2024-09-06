def parse_element(element_string):
    content = element_string.split(' ')
    ret_dict = {}
    ret_dict['type'] = content[0][0]
    ret_dict['suffix'] = content[0][1:]
    ret_dict['p_node'] = int(content[1])
    ret_dict['n_node'] = int(content[2])
    ret_dict['value'] = float(content[3])
    return ret_dict

def read_netlist(file_name):
    f = open(file_name, 'r')
    content = f.read()
    content = content.splitlines()
    content = list(map(parse_element, content))
    print(content)

