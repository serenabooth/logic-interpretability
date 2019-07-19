import circuits
import networkx as nx
import re

def read_file(fname):
    """
    Open a file to read.

    Parameters
    ----------
    fname : str
        Name of file

    Returns
    -------
    fp
        File opened to read
    """
    try:
        fp = open(fname, 'r')
        return fp
    except IOError:
        print ("Could not read file", fname)
        sys.exit()

def write_file(fname):
    """
    Open a file to write.

    Parameters
    ----------
    fname : str
        Name of file

    Returns
    -------
    fp
        File opened to write
    """
    try:
        fp = open(fname, 'w')
        return fp
    except IOError:
        print ("Could not write file", fname)
        sys.exit()


def parse_NNF_DIMACS(fname):
    assert(fname.endswith(".nnf")), "File not .nnf format"
    """
    Parse a DIMACS .nnf file.

    The .nnf format is specified in the included file /FormulaFormats.pdf.
    The original source is http://reasoning.cs.ucla.edu/c2d/, from Adnan Darwiche's lab.

    Parameters
    ----------
    fname : str
        Name of file storing an Circuit sentence

    Returns
    -------
    nnf
        An Circuit representing the DIMACs file contents

    """
    nnf_dimacs_form = read_file(fname)
    idx = 0

    all_variables = set()
    graph_representation = circuits.DiGraph()
    clauses = []

    # todo - add clauses to Circuit

    for line in nnf_dimacs_form:
        line_payload = line.split()[:]
        # ignore comments
        if line_payload[0] == "c":
            pass
        # get topline data
        elif line_payload[0] == "nnf":
            (nNodes, nEdges, nVars) = map(int, line_payload[1:])
        # add a leaf
        elif line_payload[0] == "L":
            variable = int(line_payload[1])
            all_variables.add(abs(variable))
            graph_representation.add_node(idx, value=variable, type="LEAF")
            idx += 1
        # add an and clause
        elif line_payload[0] == "A":
            num_child_nodes = int(line_payload[1])
            # A 0 indicates a "True" leaf
            if num_child_nodes == 0:
                graph_representation.add_node(idx, value="True", type="LEAF")
                idx += 1
            else:
                child_nodes = map(int, line_payload[2:])
                graph_representation.add_node(idx, value="and", type="LOGIC_GATE")
                for child in child_nodes:
                    graph_representation.add_edge(idx, child)
                # append the index of the AND node in the graph to the list of clauses
                clauses.append(idx)
                idx += 1
        # add an or clause
        elif line_payload[0] == "O":
            conflict_variable = int(line_payload[1])
            #TODO: handle conflict variable
            num_child_nodes = int(line_payload[2])
            child_nodes = map(int, line_payload[3:])
            # O j 0 indicates a "False" leaf
            if num_child_nodes == 0:
                graph_representation.add_node(idx, value="False", type="LEAF")
                idx += 1
            else:
                graph_representation.add_node(idx, value="or", type="LOGIC_GATE")
                for child in child_nodes:
                    graph_representation.add_edge(idx, child)
                # append the index of the OR node in the graph to the list of clauses
                clauses.append(idx)
                idx += 1

    # Specifically for dDNNF conversions: some add an extraneous "True" or "False"
    # node. Remove this.
    roots = []
    for node in graph_representation.nodes():
        if graph_representation.in_degree(node) == 0:
            roots.append(node)
    if len(roots) > 1:
        for root in roots:
            if graph_representation.out_degree(root) == 0:
                graph_representation.remove_node(root)

    # If the graph is not a DAG, return None
    roots = []
    for node in graph_representation.nodes():
        if graph_representation.in_degree(node) == 0:
            roots.append(node)
    if len(roots) > 1 or len(roots) == 0:
        print ("RETURNING NONE")
        return None

    nnf = circuits.Circuit(clauses=clauses,
        variables=all_variables,
        num_edges=nEdges,
        num_nodes = nNodes,
        graph_representation = graph_representation)

    nnf_dimacs_form.close()
    return nnf


def parse_CNF_DIMACS(fname):
    assert(fname.endswith(".cnf")), "File not .cnf format"
    """
    Parse a DIMACS .cnf file.

    The .cnf format is specified in the included file /FormulaFormats.pdf.
    One source is http://reasoning.cs.ucla.edu/c2d/, from Adnan Darwiche's lab.
    Note that CNF is a subset of NNF.
    As such, read in NNF-like format (with assigned indices for each node).

    Parameters
    ----------
    fname : str
        Name of file storing a CNF sentence

    Returns
    -------
    nnf
        An NNF representing the DIMACs file contents

    """
    cnf_dimacs_form = read_file(fname)

    idx = 0

    all_variables = set()
    graph_representation = circuits.DiGraph()
    clauses = []

    for line in cnf_dimacs_form:
        line_payload = line.split()[:]
        # ignore comments
        if line_payload[0] == "c":
            pass
        # get topline data
        elif line_payload[0] == "p" and line_payload[1] == "cnf":
            (nVars, nClauses) = map(int, line_payload[2:])
        # add each clause
        elif line_payload[0].replace('-','').isdigit():
            # track the added leaves for each clause for joining OR node
            added_leaves = []
            # create a LEAF node for each mentioned variable
            for variable in map(int, line_payload[0:-1]):
                all_variables.add(abs(variable))
                graph_representation.add_node(idx, value=variable, type="LEAF")
                added_leaves.append(idx)
                idx += 1
            # create an OR node joining all variables listed in a clause
            graph_representation.add_node(idx, value="or", type="LOGIC_GATE")
            for leaf in added_leaves:
                graph_representation.add_edge(idx, leaf)
            clauses.append(idx)

            idx +=1

    # if graph is empty, just return it like that
    if len(graph_representation.nodes()) == 0:
        print ("GRAPH WAS EMPTY")
        return None

    # otherwise, join all OR nodes (tracked as clauses) with an AND node
    graph_representation.add_node(idx, value="and", type="LOGIC_GATE")
    for clause in clauses:
        graph_representation.add_edge(idx, clause)
    clauses.append(idx)

    nnf = circuits.Circuit(clauses = clauses,
        variables = all_variables,
        num_edges = graph_representation.number_of_nodes(),
        num_nodes = graph_representation.number_of_edges(),
        graph_representation = graph_representation)

    cnf_dimacs_form.close()
    return nnf


def parse_DNF_natural_language(dnf_text_description):
    """
    Given a DNF expression in natural language form, convert it to a DIMACS file.

    Example strings:
        - ((1 and 2) or (-3) or (-4 and 5))
        - ((1 and 2 and 3))
    CAUTION: formatting of input strings is strict.
    """
    graph_representation = circuits.DiGraph()
    idx = 1
    or_node = idx
    graph_representation.add_node(or_node, value="or", type="LOGIC_GATE")
    idx += 1

    dnf_text_description = dnf_text_description[1:-1]

    while len(dnf_text_description) > 0:
        if dnf_text_description.startswith( '(' ):
            and_node = idx
            idx += 1
            graph_representation.add_node(and_node, value="and", type="LOGIC_GATE")
            graph_representation.add_edge(or_node, and_node)
            substring_end = dnf_text_description.find( ")" )
            leaves = re.findall('-?\d+',dnf_text_description[:substring_end])

            for leaf in leaves:
                graph_representation.add_node(idx, value=int(leaf), type="LEAF")
                graph_representation.add_edge(and_node, idx)
                idx += 1
            dnf_text_description = dnf_text_description[substring_end+1:]
        if dnf_text_description.startswith( ' or ' ):
            dnf_text_description = dnf_text_description[len( ' or ' ):]

    clauses = [node for node in graph_representation.nodes() if graph_representation.nodes[node]['type'] == 'LOGIC_GATE']
    circuit = circuits.Circuit(
        clauses = clauses,
        variables = graph_representation.get_variables(graph_representation.find_dag_root()),
        num_edges = graph_representation.number_of_nodes(),
        num_nodes = graph_representation.number_of_edges(),
        graph_representation = graph_representation)

    return circuit


def write_NNF_DIMACS(nnf, fname):
    assert nnf.is_nnf(), "Can only write NNF's. Please make NNF first"
    assert(fname.endswith(".nnf")), "File not .nnf format"
    """
    Write a .nnf file for the contents of an NNF object

    Parameters
    ----------
    nnf : NNF
        an NNF with an instantiated DAG, representing a logical statement
    fname : str
        name of file to write out to; must be *.nnf
    """
    nnf_file = write_file(fname)

    # write header: nnf n m v
    root = nnf.graph.find_dag_root()
    nNodes = len(nnf.graph.nodes())
    nEdges = len(nnf.graph.edges())
    nVariables = len(nnf.graph.get_variables(root))
    nnf_file.write("nnf " + str(nNodes) + " " + str(nEdges) + " " + str(nVariables) + "\n")

    # find highest variable identifier
    max_value = max([abs(value) for _, value in
                        nx.get_node_attributes(nnf.graph, 'value').items()
                        if isinstance(value, int)])
    max_value += 1
    node_indeces_to_line_indeces = {}
    idx = 0

    nodes = set(nnf.graph.nodes())
    leaves = nnf.graph.find_all_leaves()

    converted_aux_variables = {}

    # record the leaves
    for leaf in leaves:
        # check if alphanumeric
        if isinstance(nnf.graph.nodes[leaf]['value'], int):
            nnf_file.write("L " + str(nnf.graph.nodes[leaf]['value']) + "\n")
        else:
            # replace with highest int not used
            variable_value =  nnf.graph.node[leaf]['value'].replace("-","")
            if nnf.graph.node[leaf]['value'][0] == "-" and variable_value in converted_aux_variables:
                nnf_file.write("L -" + str(converted_aux_variables[variable_value]) + "\n")
            elif variable_value in converted_aux_variables:
                nnf_file.write("L " + str(converted_aux_variables[variable_value]) + "\n")
            else:
                nnf_file.write("L " + str(max_value) + "\n")
                converted_aux_variables[variable_value] = max_value
                max_value += 1

        node_indeces_to_line_indeces[leaf] = idx
        idx += 1
        nodes.remove(leaf)

    while len(nodes) > 0:
        # evaluate whether each node can be recorded and removed
        nodes_to_remove = set()

        # iterate through nodes; write those whose children have already been recorded
        for node in nodes:
            # remaining nodes should not be leaves
            if nnf.graph.nodes[node]['type'] == "LEAF":
                raise Exception("Remaining nodes should not be leaves")

            # check all children have been recorded
            write_node = True
            for child in nnf.graph.neighbors(node):
                if child not in node_indeces_to_line_indeces.keys():
                    write_node = False

            if write_node:
                if nnf.graph.nodes[node]['value'] == "or":
                    is_decision_node, decision_node_values = nnf.is_decision_node(node)
                    if is_decision_node and len(decision_node_values) > 0:
                        nnf_file.write("O " + str(abs(decision_node_values[0])) + " ")
                    else:
                        nnf_file.write("O 0 ")
                elif nnf.graph.nodes[node]['value'] == "and":
                    nnf_file.write("A ")

                nnf_file.write(str(nnf.graph.out_degree(node)) + " ")
                for child in nnf.graph.neighbors(node):
                    nnf_file.write(str(node_indeces_to_line_indeces[child]) + " ")
                nnf_file.write("\n")

                node_indeces_to_line_indeces[node] = idx
                idx += 1

                nodes_to_remove.add(node)

        nodes = nodes.difference(nodes_to_remove)
    nnf_file.close()


def write_CNF_DIMACS(nnf, fname):
    assert(fname.endswith(".cnf")), "File not .cnf format"
    assert circuits.Languages.CNF in nnf.languages, "Must be a CNF formula"
    """
    Write a .cnf file for the contents of an NNF object

    Parameters
    ----------
    nnf : NNF
        an NNF with an instantiated DAG, representing a logical statement
        must meet CNF criteria
    fname : str
        name of file to write out to; must be *.cnf
    """
    nnf_file = write_file(fname)

    root = nnf.graph.find_dag_root()
    max_value = max([abs(value) for _, value in
                        nx.get_node_attributes(nnf.graph, 'value').items()
                        if isinstance(value, int)])
    nVariables = max_value
    max_value += 1
    converted_aux_variables = {}

    #nVariables = len(nnf.graph.get_variables(root))
    nClauses = 0
    for clause in nnf.clauses:
        if nnf.graph.node[clause]['value'] == "or":
            nClauses += 1
    for child in nnf.graph.neighbors(root):
        if nnf.graph.node[child]['type'] == "LEAF":
            nClauses += 1

    # if DAG consists of only one node
    if nnf.graph.node[root]['type'] == "LEAF":
        nnf_file.write("p cnf 1 1\n")
        if isinstance(nnf.graph.nodes[root]['value'], int):
            nnf_file.write(str(nnf.graph.nodes[root]['value']) + " 0\n")
        else:
            variable_value = nnf.graph.node[root]['value'].replace("-","")
            if nnf.graph.node[root]['value'][0] == "-" and variable_value in converted_aux_variables:
                nnf_file.write(str(converted_aux_variables[variable_value]) + " 0\n")
            elif variable_value in converted_aux_variables:
                nnf_file.write(str(converted_aux_variables[variable_value]) + " 0\n")
            else:
                nnf_file.write(str(max_value) + " 0\n")
                converted_aux_variables[variable_value] = max_value
                max_value += 1
        return

    # DAG consists of more than one node
    nnf_file.write("p cnf " + str(nVariables) + " " + str(nClauses) + "\n")
    for child in nnf.graph.neighbors(root):
        if nnf.graph.node[child]['type'] == "LEAF":
            if isinstance(nnf.graph.nodes[child]['value'], int):
                nnf_file.write(str(nnf.graph.nodes[child]['value']) + " 0\n")
            else:
                variable_value =  nnf.graph.node[child]['value'].replace("-","")
                if nnf.graph.node[root]['value'][0] == "-" and variable_value in converted_aux_variables:
                    nnf_file.write(str(converted_aux_variables[variable_value]) + " 0\n")
                elif variable_value in converted_aux_variables:
                    nnf_file.write(str(converted_aux_variables[variable_value]) + " 0\n")
                else:
                    nnf_file.write(str(max_value) + " 0\n")
                    converted_aux_variables[variable_value] = max_value
                    max_value += 1

        elif nnf.graph.node[child]['value'] == "or":
            for grandchild in nnf.graph.neighbors(child):
                if nnf.graph.node[grandchild]['type'] == "LEAF":
                    if isinstance(nnf.graph.nodes[grandchild]['value'], int):
                        nnf_file.write(str(nnf.graph.nodes[grandchild]['value']) + " ")
                    else:
                        variable_value = nnf.graph.node[grandchild]['value'].replace("-","")
                        if nnf.graph.node[grandchild]['value'][0] == "-" and variable_value in converted_aux_variables:
                            nnf_file.write(str(converted_aux_variables[variable_value]) + " ")
                        elif variable_value in converted_aux_variables:
                            nnf_file.write(str(converted_aux_variables[variable_value]) + " ")
                        else:
                            nnf_file.write(str(max_value) + " ")
                            converted_aux_variables[variable_value] = max_value
                            max_value += 1
            nnf_file.write("0\n")

        elif nnf.graph.node[child]['value'] == "and":
            raise Exception("And node should not have and children in CNF form")
    nnf_file.close()
