import networkx as nx
import itertools
import copy
import subprocess
import time
import sys
import re
from pathlib import Path

import circuits
import circuit_io

from pyeda.boolalg.bdd import *
from pyeda.inter import *

DEBUG = True

def compute_clauses(graph):
    """
    Given an arbitrary graph (representing a circuit), find the nodes corresponding to logic gates

    Parameters
    ----------
    graph : DiGraph
        An instantiated graph representation of a logical sentence

    Returns
    -------
    list of integers
        List of integers corresponding to logic gates indices in graph
    """
    clauses = []
    for node in graph.nodes():
        if graph.node[node]['type'] == "LOGIC_GATE":
            clauses.append(node)
    return clauses

def bdd_expr_to_dag(bdd, idx, variable_mapping):
    """
    Convert a BDD (denoted in pyeda syntax) to a graph.

    Parameters
    ----------
    bdd : String
        Example: "Or(And(a,b,c),And(-a,b))"
    idx : int
        Used as a counter to add indexed nodes to the graph
    variable_mapping : dict
        Maps strings(e.g. "a") to integers (e.g. 1)
        DIMACs stores variables as integers; pyeda uses strings
    """
    graph_representation = circuits.DiGraph()
    if bdd[0:3] == "Or(":
        paren_count = 1

        root = idx
        graph_representation.add_node(root, value="or", type="LOGIC_GATE")
        idx += 1

        substring_start = 3
        for str_idx in range(substring_start, len(bdd)):
            if bdd[str_idx] == "(":
                paren_count += 1
            elif bdd[str_idx] == ")":
                paren_count -= 1

            if paren_count == 1 and bdd[str_idx] == "," or \
                    paren_count == 0 and str_idx == len(bdd)-1:
                subgraph, max_idx = bdd_expr_to_dag(bdd[substring_start:str_idx], idx, variable_mapping)
                graph_representation = nx.compose(graph_representation, subgraph)
                subgraph_root = subgraph.find_dag_root()
                graph_representation.add_edge(root, subgraph_root)
                idx = max(idx, max_idx)
                substring_start = str_idx+1

    elif bdd[0:4] == "And(":
        paren_count = 1
        root = idx
        graph_representation.add_node(idx, value="and", type="LOGIC_GATE")
        idx += 1

        substring_start = 4
        for str_idx in range(substring_start, len(bdd)):
            if bdd[str_idx] == "(":
                paren_count += 1
            elif bdd[str_idx] == ")":
                paren_count -= 1

            if (paren_count == 1 and bdd[str_idx] == ",") or \
                    paren_count == 0 and str_idx == len(bdd)-1:
                subgraph, max_idx = bdd_expr_to_dag(bdd[substring_start:str_idx], idx, variable_mapping)
                graph_representation = nx.compose(graph_representation, subgraph)
                subgraph_root = subgraph.find_dag_root()
                graph_representation.add_edge(root, subgraph_root)
                idx = max(idx, max_idx)
                substring_start = str_idx+1
    else:
        if bdd[0] == "-":
            graph_representation.add_node(idx, value=-1 * variable_mapping[bdd[1:]], type="LEAF")
        else:
            graph_representation.add_node(idx, value=variable_mapping[bdd], type="LEAF")
        idx += 1

    return (graph_representation, idx)


# TODO - would be nice to write own version of BDD construction code
def convert_nnf_to_bdd(circuit):
    """
    Using the pyeda library, converts an arbitrary circuit to BDD representation
    """
    root = circuit.graph.find_dag_root()
    variables = list(circuit.graph.get_variables(root))

    conv_string = "abcdefghij"

    # create map of variables (e.g. 1, 2) to strings (e.g. "a", "b")
    bdd_variables = {}
    # create reverse map (e.g. "a", "b" to 1, 2)
    reverse_mapping_variables_bdd = {}
    for i in range(0, len(variables)):
        number_to_string = str(variables[i])
        tmp_string = ""
        for char in number_to_string:
            tmp_string += conv_string[int(char)]

        bdd_variables[variables[i]] = bddvar(tmp_string)
        reverse_mapping_variables_bdd[tmp_string] = variables[i]

    expression = circuit.graph.collapse_graph_to_formula_promela_syntax(root)

    for variable in variables:
        expression = expression.replace(str(variable), str(bdd_variables[variable]))

    bdd = expr2bdd(expr(expression))
    bdd = bdd2expr(bdd)
    bdd = str(bdd).replace(" ", "").replace("~", "-")

    graph, _ = bdd_expr_to_dag(bdd, 0, reverse_mapping_variables_bdd)
    return circuits.Circuit(clauses = compute_clauses(graph),
                   variables = graph.get_variables(graph.find_dag_root()),
                   num_edges = len(graph.edges()),
                   num_nodes = len(graph.nodes()),
                   graph_representation = graph)


def convert_to_nnf_helper(circuit):
    """
    Converts the current circuit into one that is negation normal form (NNF)

    Parameters
    ----------
    circuit : Circuit
        Circuit instantiation with graph_representation

    Returns
    -------
    nnf_representation : DiGraph
        A directed acyclic graph corresponding to an NNF representation
    """
    circuit = copy.deepcopy(circuit)
    changed = True
    flips = 0
    while changed:
        changed = False
        for (nid, node) in circuit.graph.nodes().items():
            if node['value'] in ['nand','nor']:
                flips += 1
                # Go around the loop again (we could be smarter about this if we wanted)
                changed = True
                # Negate the parent
                circuit._negate_node(node)
                # Negate the children
                for n in circuit.graph.neighbors(nid):
                    circuit._negate_node(circuit.graph.nodes()[n])

    if DEBUG:
        print("Made %d flips for enforcing the NNF property." % flips)

    return circuit.graph


def convert_to_nnf(circuit):
    """
    Convert an arbitrary logical sentence to an NNF sentence

    Parameters
    ----------
    circuit : Circuit
        An arbitrary circuit with an instantiated graph representation

    Returns
    -------
    Circuit
        An equivalent NNF reformatted Circuit statement
    """
    nnf_graph = convert_to_nnf_helper(circuit)
    return circuits.Circuit(clauses = compute_clauses(nnf_graph),
                   variables = nnf_graph.get_variables(nnf_graph.find_dag_root()),
                   num_edges = len(nnf_graph.edges()),
                   num_nodes = len(nnf_graph.nodes()),
                   graph_representation = nnf_graph)


def convert_to_epsinv_helper(circuit):
    """
    Converts the current circuit into one where the epsilon inverted property holds.

    Conjecture: This will not face infinite cycles.

    Parameters
    ----------
    circuit : Circuit
        Circuit instantiation with graph_representation

    Returns
    -------
    graph : DiGraph
        DiGraph representing an epsilon-inverted statement
    """
    # do not modify original circuit
    circuit = copy.deepcopy(circuit)
    changed = True
    flips = 0
    while changed:
        changed = False
        for (nid, node) in circuit.graph.nodes().items():
            if not circuit._is_epsilon_inverted(nid, node):
                flips += 1
                # Go around the loop again (we could be smarter about this if we wanted)
                changed = True
                # Negate the parent
                circuit._negate_node(node)
                # Negate the children
                for n in circuit.graph.neighbors(nid):
                    circuit._negate_node(circuit.graph.nodes()[n])

    if DEBUG:
        print("Made %d flips for enforcing the epsilon inverted property." % flips)

    return circuit.graph


def convert_to_epsinv(circuit):
    """
    Convert an arbitrary logical sentence to an epsilon-inverted sentence

    Parameters
    ----------
    circuit : Circuit
        An arbitrary circuit with an instantiated graph representation

    Returns
    -------
    Circuit
        An equivalent epsilon-inverted reformatted Circuit statement
    """
    e_graph = convert_to_epsinv_helper(circuit)
    return circuits.Circuit(clauses = compute_clauses(e_graph),
                   variables = e_graph.get_variables(e_graph.find_dag_root()),
                   num_edges = len(e_graph.edges()),
                   num_nodes = len(e_graph.nodes()),
                   graph_representation = e_graph)


def convert_to_cnf_helper(nnf, clause_idx, idx):
    """
    Apply the rules of distribution, double negation, and de morgan's
    law to convert any formula to CNF.

    NOTE: This is NOT the Tseitin transformation, and does not
    introduce auxilary variables as such.

    https://www.cs.jhu.edu/~jason/tutorials/convert-to-CNF.html

    Parameters
    ----------
    nnf : Circuit
        Circuit instantiation with graph_representation

    clause_idx : int
        An integer identifying a node in the given NNF

    idx : int
        A running counter to add new nodes to the graph
        (node identifier must be unique)

    Returns
    -------
    cnf_representation : DiGraph
        A directed acyclic graph corresponding to a CNF
        representation of the passed in NNF
    """

    # If phi is a variable, then return phi.
    # this is a CNF formula consisting of 1 clause that contains 1 literal
    if nnf.graph.node[clause_idx]['type'] == "LEAF":
        return nnf.graph.subgraph(clause_idx)

    # If phi has the form P ^ Q, then:
    # return P1 ^ P2 ^ ... ^ Pm ^ Q1 ^ Q2 ^ ... ^ Qn.
    elif nnf.graph.node[clause_idx]['value'] == "and":
        subgraph_roots = []
        cnf_representation = circuits.DiGraph()

        for child in nx.neighbors(nnf.graph, clause_idx):
            idx = max(idx, max(cnf_representation.nodes() or [-1])) + 1

            # recursively compute the CNF representation for each child clause
            subgraph_to_add = convert_to_cnf_helper(nnf, child, idx)
            idx = max(idx, max(subgraph_to_add.nodes() or [-1])) + 1
            subgraph_root = subgraph_to_add.find_dag_root()

            # case: an AND node pointing to an AND node
            # remove the existing AND node and create a new one
            if subgraph_to_add.node[subgraph_root]['value'] == "and":
                node_descendants = nx.descendants(subgraph_to_add, subgraph_root)
                for node in nx.neighbors(subgraph_to_add, subgraph_root):
                    subgraph_roots.append(node)
                subgraph_to_add = nx.subgraph(subgraph_to_add, node_descendants)
            else:
                subgraph_roots.append(subgraph_root)

            cnf_representation = nx.compose(cnf_representation, subgraph_to_add)

        # create a new AND node, and connect it to the root of each subgraph
        # each subgraph root should be a literal or disjunct of literals
        cnf_representation.add_node(idx, value="and", type="LOGIC_GATE")
        for node_id in subgraph_roots:
            cnf_representation.add_edge(idx, node_id)

        return cnf_representation

    # If phi has the form P v Q, then:
    # return (P1 v Q1) ^ (P1 v Q2) ^ ... ^ (P1 v Qn)
    #        ^ (P2 v Q1) ^ (P2 v Q2) ^ ... ^ (P2 v Qn)
    #          ...
    #        ^ (Pm v Q1) ^ (Pm v Q2) ^ ... ^ (Pm v Qn)
    elif nnf.graph.node[clause_idx]['value'] == "or":
        # For P v Q, create a list of lists of the form
        # [[P1, P2, ..., Pm], [Q1, Q2, ..., Qn]]
        subgraphs = []
        cnf_representation = circuits.DiGraph()

        for child in nnf.graph.neighbors(clause_idx):
            idx = max(idx, max(cnf_representation.nodes() or [-1])) + 1

            # recursively compute the CNF representation for each child clause
            subgraph_to_add = convert_to_cnf_helper(nnf, child, idx)
            idx = max(idx, max(subgraph_to_add.nodes() or [-1])) + 1
            subgraph_root = subgraph_to_add.find_dag_root()

            subgraph = []
            if subgraph_to_add.node[subgraph_root]['value'] == "and":
                for grandchild in nx.neighbors(subgraph_to_add, subgraph_root):
                    subgraph.append(grandchild)

                node_descendants = nx.descendants(subgraph_to_add, subgraph_root)
                subgraph_to_add = nx.subgraph(subgraph_to_add, node_descendants)
            else:
                subgraph.append(subgraph_root)

            subgraphs.append(subgraph)

            cnf_representation = nx.compose(cnf_representation, subgraph_to_add)

        combinations_of_disjuncts = list(itertools.product(*subgraphs))

        cnf_representation.add_node(idx, value="and", type="LOGIC_GATE")
        and_index = idx

        nodes_to_remove = set()
        for disjuncts in combinations_of_disjuncts:
            idx += 1
            cnf_representation.add_node(idx, value="or", type="LOGIC_GATE")
            cnf_representation.add_edge(and_index, idx)
            for individual_node in disjuncts:
                # if an OR node already, combine
                if cnf_representation.node[individual_node]['value'] == "or":
                    for child in nx.neighbors(cnf_representation, individual_node):
                        cnf_representation.add_edge(idx, child)
                    nodes_to_remove.add(individual_node)
                # else, add edges from the new OR node to the individual nodes
                else:
                    cnf_representation.add_edge(idx, individual_node)

        for node in nodes_to_remove:
            cnf_representation.remove_node(node)

        return cnf_representation

def convert_to_cnf(nnf):
    """
    Convert an NNF to an NNF which is a member of CNF

    Parameters
    ----------
    nnf : Circuit
        An arbitrary NNF statement with an instantiated graph representation

    Returns
    -------
    Circuit
        An equivalent CNF reformatted Circuit statement
    """
    assert nnf.is_nnf(), "Can only convert NNF's to CNF. Please make NNF first"
    root = nnf.graph.find_dag_root()
    assert(root > -1), "No root found"

    cnf_graph = convert_to_cnf_helper(nnf, root, max(nnf.graph.nodes()))

    return circuits.Circuit(clauses = compute_clauses(cnf_graph),
                   variables = cnf_graph.get_variables(cnf_graph.find_dag_root()),
                   num_edges = len(cnf_graph.edges()),
                   num_nodes = len(cnf_graph.nodes()),
                   graph_representation = cnf_graph)


def tseitin_helper(nnf, root, clause_idx, idx):
    """
    Apply the rules of Tseitin to transform an NNF to CNF.

    Parameters
    ----------
    nnf : NNF
        An NNF representation of a logic statement, with
        an instantiated DAG
    root : int
        An integer indicating the root of the graph
    clause_idx : int
        An integer indicating the current node of observation
    idx : int
        An integer used to track the highest node index

    Returns
    -------
    DiGraph

    """
    cnf_representation = circuits.DiGraph()

    # create the parent AND node
    cnf_representation.add_node(idx, value="and", type="LOGIC_GATE")
    parent_and_idx = idx
    idx += 1

    # add a clause consisting of just the variable of the root node (tseitin id)
    if (clause_idx == root):
        node = nnf.graph.node[clause_idx]
        if node['type'] == "LEAF":
            return nnf.graph.subgraph(clause_idx)
        else:
            cnf_representation.add_node(idx, value=node['tseitin_aux_var'], type="LEAF", aux_var=True)
            cnf_representation.add_edge(parent_and_idx, idx)
            idx += 1


    # if node is a leaf, we're done; return an empty graph
    if nnf.graph.node[clause_idx]['type'] == "LEAF":
        return circuits.DiGraph()

    # if the node is a LOGIC_GATE, add a node representing it with an aux variable
    # create "a" node
    cnf_representation.add_node(idx, value=nnf.graph.node[clause_idx]['tseitin_aux_var'],
                                     type="LEAF", aux_var=True)
    this_node_idx = idx
    idx += 1
    # create "-a" node
    cnf_representation.add_node(idx, value="-" + nnf.graph.node[clause_idx]['tseitin_aux_var'],
                                     type="LEAF", aux_var=True)
    this_node_negated_idx = idx
    idx += 1

    children = nnf.graph.neighbors(clause_idx)
    list_of_child_nodes = []
    list_of_child_nodes_negated = []

    # create a copy of each child node, and its negated self in cnf_representation
    for child in children:
        if nnf.graph.node[child]['type'] == "LEAF":
            cnf_representation.add_node(child, value=nnf.graph.node[child]['value'],
                                     type=nnf.graph.node[child]['type'])
            cnf_representation.add_node(idx, value=-1 * nnf.graph.node[child]['value'],
                                     type=nnf.graph.node[child]['type'])
            list_of_child_nodes.append(child)
            list_of_child_nodes_negated.append(idx)
            idx += 1
        else:
            cnf_representation.add_node(idx, value=nnf.graph.node[child]['tseitin_aux_var'],
                                     type="LEAF", aux_var=True)
            list_of_child_nodes.append(idx)
            idx += 1
            cnf_representation.add_node(idx, value="-" + nnf.graph.node[child]['tseitin_aux_var'],
                                     type="LEAF", aux_var=True)
            list_of_child_nodes_negated.append(idx)
            idx += 1

    # if node is of the form b = formula_1, a = and, c = formula_2,
    # add clauses (not a or b) and (not a or c) and (not b or not c or a)
    if nnf.graph.node[clause_idx]['value'] == "and":
        # add all terms of the form (not a or b)
        for i in range(0, len(list_of_child_nodes)):
            cnf_representation.add_node(idx, value="or", type="LOGIC_GATE")
            cnf_representation.add_edge(parent_and_idx, idx)
            # construct the ORs
            cnf_representation.add_edge(idx, this_node_negated_idx)
            cnf_representation.add_edge(idx, list_of_child_nodes[i])
            idx += 1

        # add the term of the form (a or not b or not c)
        cnf_representation.add_node(idx, value="or", type="LOGIC_GATE")
        cnf_representation.add_edge(parent_and_idx, idx)
        or_node = idx
        idx +=  1
        cnf_representation.add_edge(or_node, this_node_idx)
        for i in range(0, len(list_of_child_nodes_negated)):
            cnf_representation.add_edge(or_node, list_of_child_nodes_negated[i])

    # if node is of the form b = formula_1, a = or, c = formula_2,
    # add clauses (not a or b or c) and (not b or a) and (not c or a)
    if nnf.graph.node[clause_idx]['value'] == "or":
        # add all terms of the form (not b or a)
        for i in range(0, len(list_of_child_nodes_negated)):
            cnf_representation.add_node(idx, value="or", type="LOGIC_GATE")
            cnf_representation.add_edge(parent_and_idx, idx)
            # construct the ORs
            cnf_representation.add_edge(idx, this_node_idx)
            cnf_representation.add_edge(idx, list_of_child_nodes_negated[i])
            idx += 1

        # add node of the form (not a or b or c)
        cnf_representation.add_node(idx, value="or", type="LOGIC_GATE")
        cnf_representation.add_edge(parent_and_idx, idx)
        or_node = idx
        idx +=  1
        cnf_representation.add_edge(or_node, this_node_negated_idx)
        for i in range(0, len(list_of_child_nodes)):
            cnf_representation.add_edge(or_node, list_of_child_nodes[i])

    # recursively add clauses for child nodes
    for child in nnf.graph.neighbors(clause_idx):
        child_graph = tseitin_helper(nnf, root, child, idx)
        idx = max(idx, max(child_graph.nodes() or [-1])) + 1

        if len(child_graph.nodes()) > 0:
            cnf_representation = nx.compose(cnf_representation, child_graph)

            child_root = child_graph.find_dag_root()
            # if top of child node is an "AND", combine with the current parent
            if child_graph.node[child_root]['value'] == "and":
                for grandchild in child_graph.neighbors(child_root):
                    cnf_representation.add_edge(parent_and_idx, grandchild)
                cnf_representation.remove_node(child_root)
            else:
                cnf_representation.add_edge(parent_and_idx, child_root)

    return cnf_representation


def tseitin(nnf):
    """
    Convert an NNF to an NNF which is a member of CNF
    using Tseitin Translation

    https://profs.info.uaic.ro/~stefan.ciobaca/logic-2018-2019/notes7.pdf


    Parameters
    ----------
    nnf : NNF
        An arbitrary NNF statement with an instantiated graph representation

    Returns
    -------
    NNF
        An equivalent CNF reformatted NNF statement
    """
    assert nnf.is_nnf(), "Can only convert NNF's to CNF. Please make NNF first"
    root = nnf.graph.find_dag_root()
    assert(root > -1), "No root found"

    idx = max(nnf.graph.nodes()) + 1

    # associate a new variable with each LOGIC_GATE node in the DAG
    for i in range(0, len(nnf.clauses)):
        nnf.graph.node[nnf.clauses[i]]['tseitin_aux_var'] = "t_" + str(idx)
        idx += 1

    cnf_graph = tseitin_helper(nnf, root, root, idx)

    return circuits.Circuit(clauses = compute_clauses(cnf_graph),
                   variables = cnf_graph.get_variables(cnf_graph.find_dag_root()),
                   num_edges = len(cnf_graph.edges()),
                   num_nodes = len(cnf_graph.nodes()),
                   graph_representation = cnf_graph)


def convert_ddnnf_to_odnf_helper(circuit, clause_idx):
    """
    Given a ddnnf form, create the schema required to convert it to d-DNNF.

    Parameters
    ----------
    circuit : Circuit
        With instantiated DAG
    clause_idx : int
        Pointer to the current node under consideration

    Returns
    -------
    list of lists
        For example [[1,2,3],[-2,3]] represents the ODNF
        (1 and 2 and 3) or (-2 and 3)
    """
    nodes = []
    if circuit.graph.node[clause_idx]['type'] == "LEAF":
        return [clause_idx]
    elif circuit.graph.node[clause_idx]['value'] == "or":
        for child in circuit.graph.neighbors(clause_idx):
            tmp = convert_ddnnf_to_odnf_helper(circuit, child)
            if (isinstance(tmp[0], list)):
                nodes += tmp
            else:
                nodes.append(tmp)
        return nodes
    else: # node is AND
        num_children = 0
        for child in circuit.graph.neighbors(clause_idx):
            num_children += 1
            nodes.append(convert_ddnnf_to_odnf_helper(circuit, child))

        # combine all of the elements with the first one
        # for example, [[1,2], [3,4], [5]] should give [[1,2,3,4,5]]
        # (order doesn't matter)
        for i in range(1, num_children):
            if isinstance(nodes[0][0], list):
                for node in nodes[0]:
                    node += nodes[i]
            else:
                nodes[0] += nodes[i]

        return nodes[0]


def convert_ddnnf_to_odnf(circuit):
    """
    Given a ddnnf form, convert it to ODNF.

    Parameters
    ----------
    circuit : Circuit
        With instantiated DAG

    Returns
    -------
    circuit in ODNF form
    """
    assert circuit.is_decomposable() and circuit.is_deterministic(), "Can only convert dDNNF to ODNF"
    root = circuit.graph.find_dag_root()
    odnf_graph_schema = convert_ddnnf_to_odnf_helper(circuit, root)

    odnf_graph = circuits.DiGraph()
    idx = max(circuit.graph.nodes()) + 1
    or_node = idx
    idx += 1
    odnf_graph.add_node(or_node, value="or", type="LOGIC_GATE")
    for sublist in odnf_graph_schema:
        and_idx = idx
        odnf_graph.add_node(and_idx, value="and", type="LOGIC_GATE")
        odnf_graph.add_edge(or_node, and_idx)
        idx += 1
        for element in sublist:
            odnf_graph.add_node(idx, value=circuit.graph.node[element]['value'], type=circuit.graph.node[element]['type'])
            odnf_graph.add_edge(and_idx, idx)
            idx += 1

    return circuits.Circuit(clauses = compute_clauses(odnf_graph),
                   variables = odnf_graph.get_variables(or_node),
                   num_edges = len(odnf_graph.edges()),
                   num_nodes = len(odnf_graph.nodes()),
                   graph_representation = odnf_graph)


def convert_odnf_to_mods_helper(circuit):
    """
    Given a odnf form, create the schema required to convert it to MODS.

    CAUTION: this function operates on values, while most similar code
    operates on indeces

    Parameters
    ----------
    circuit : Circuit
        With instantiated DAG

    Returns
    -------
    list of lists
        For example [[1,2,3],[1,-2,3]] represents the MODS
        (1 and 2 and 3) or (1 and -2 and 3)
    """
    root = circuit.graph.find_dag_root()
    variables = circuit.graph.get_variables(root)

    mods_graph_schema = []
    for child in circuit.graph.neighbors(root):
        clause = []
        clause_variables = circuit.graph.get_variables(child)
        for grandchild in circuit.graph.neighbors(child):
            clause.append(circuit.graph.node[grandchild]['value'])
        variables_not_included = variables.difference(clause_variables)

        variables_to_append = []
        for variable in variables_not_included:
            variables_to_append.append([variable,-1*variable])

        clauses = [clause]
        for i in range(0,len(variables_to_append)):
            clauses = [clause_tmp + [var] for clause_tmp in clauses for var in variables_to_append[i]]

        mods_graph_schema += clauses
    return mods_graph_schema

def convert_odnf_to_mods(circuit):
    """
    Given an ODNF form, convert it to MODS.

    Parameters
    ----------
    circuit : Circuit
        With instantiated DAG

    Returns
    -------
    circuit in MODS form
    """
    assert circuit.is_simple_conjunction() and circuit.is_deterministic(), "Can only convert ODNF to MODS"
    mods_schema = convert_odnf_to_mods_helper(circuit)

    mods_graph = circuits.DiGraph()
    idx = 0
    or_index = idx
    idx += 1
    mods_graph.add_node(or_index, value="or", type="LOGIC_GATE")
    for clause in mods_schema:
        and_index = idx
        idx += 1
        mods_graph.add_node(and_index, value="and", type="LOGIC_GATE")
        mods_graph.add_edge(or_index, and_index)
        for variable in clause:
            mods_graph.add_node(idx, value=variable, type="LEAF")
            mods_graph.add_edge(and_index, idx)
            idx += 1

    return circuits.Circuit(clauses = compute_clauses(mods_graph),
                   variables = mods_graph.get_variables(or_index),
                   num_edges = len(mods_graph.edges()),
                   num_nodes = len(mods_graph.nodes()),
                   graph_representation = mods_graph)


def simplify_cnf(circuit):
    mypath = Path().absolute()
    print(mypath)
    circuit_io.write_CNF_DIMACS(circuit, "../nnf_examples/tmp_nnf.cnf")
    subprocess.call('../code/compiled_binaries/pmc_linux -vivification -eliminateLit -litImplied -iterate=10 ../nnf_examples/tmp_nnf.cnf > ../nnf_examples/tmp_nnf_2.cnf', shell=True)
    tmp_circuit = circuit_io.parse_CNF_DIMACS("../nnf_examples/tmp_nnf_2.cnf")
    if tmp_circuit == None:
        if DEBUG:
            print("Circuit was empty, so returning input.\
                   This may be an already minimal CNF representation.")
        return circuit.simplify_cnf()
    return tmp_circuit.simplify_cnf()

def convert_nnf_to_dDNNF(circuit):
    # if the circuit is eps-inverted, convert to nnf
    circuit = convert_to_nnf(circuit)
    # convert nnf to cnf
    circuit = convert_to_cnf(circuit)
    circuit = simplify_cnf(circuit)
    # record the file
    circuit_io.write_CNF_DIMACS(circuit, "../nnf_examples/tmp_nnf_2.cnf")

    # convert to dDNNF
    subprocess.call('../code/compiled_binaries/dsharp -disableAllLits -Fnnf ../nnf_examples/nnf_conv_ddnnf.nnf ../nnf_examples/tmp_nnf_2.cnf', shell=True)
    tmp_circuit = circuit_io.parse_NNF_DIMACS("../nnf_examples/nnf_conv_ddnnf.nnf")
    # if tmp_circuit == None:
    #     if DEBUG:
    #         print ("dsharp resulted in an empty or non DAG representation")
    #     subprocess.call('../code/compiled_binaries/d4 ../nnf_examples/tmp_nnf_2.cnf -out="../nnf_examples/nnf_conv_ddnnf.nnf"', shell=True)
    #     tmp_circuit = circuit_io.parse_NNF_DIMACS("../nnf_examples/nnf_conv_ddnnf.nnf")
    #     assert tmp_circuit != None, "Yikes... Neither dsharp nor d4 working"

    return tmp_circuit

def convert_nnf_to_sdDNNF(circuit):
    # if the circuit is eps-inverted, convert to nnf
    circuit = convert_to_nnf(circuit)
    # convert nnf to cnf
    circuit = convert_to_cnf(circuit)
    circuit = simplify_cnf(circuit)
    # record the file
    circuit_io.write_CNF_DIMACS(circuit, "../nnf_examples/tmp_nnf_2.cnf")

    subprocess.call('../code/compiled_binaries/dsharp -disableAllLits -smoothNNF -Fnnf ../nnf_examples/nnf_conv_sddnnf.nnf ../nnf_examples/tmp_nnf_2.cnf', shell=True)
    return circuit_io.parse_NNF_DIMACS("../nnf_examples/nnf_conv_sddnnf.nnf")

def convert_to_given_language(circuit, target_language, invert):
    """
    Convert DNF to arbitrary KC languages

    Parameters
    ----------
    circuit : Circuit
        A DNF representation with instantiated DAG
    target_language : circuits.Languages
        A given language from the knowledge compilation map to convert to

    Returns
    -------
    A circuit corresponding to an equivalent statement to the passed in circuit, but
    translated to the target language
    """
    assert circuits.Languages.DNF in circuit.languages, "Expected DNF input circuit"

    if target_language == circuits.Languages.CNF:
        cnf = convert_to_cnf(circuit)
        cnf = simplify_cnf(cnf)
        assert circuits.Languages.CNF in cnf.languages, "Should be CNF"
        if invert:
            return convert_to_epsinv(cnf)
        return cnf

    if target_language == circuits.Languages.DNF:
        assert circuits.Languages.DNF in circuit.languages, "Should be DNF"
        if invert:
            return convert_to_epsinv(circuit)
        return circuit

    if target_language == circuits.Languages.dDNNF:
        cnf = convert_to_cnf(circuit)
        cnf = simplify_cnf(cnf)
        dDNNF = convert_nnf_to_dDNNF(cnf)
        assert circuits.Languages.dDNNF in dDNNF.languages, "Should be dDNNF"
        if invert:
            return convert_to_epsinv(dDNNF)
        return dDNNF

    if target_language == circuits.Languages.sdDNNF:
        cnf = convert_to_cnf(circuit)
        cnf = simplify_cnf(cnf)
        sdDNNF = convert_nnf_to_sdDNNF(cnf)
        assert circuits.Languages.sdDNNF in sdDNNF.languages, "Should be sdDNNF"
        if invert:
            return convert_to_epsinv(sdDNNF)
        return sdDNNF

    if target_language == circuits.Languages.ODNF:
        ODNF = convert_ddnnf_to_odnf(circuit)
        assert circuits.Languages.ODNF in ODNF.languages, "Should be ODNF"
        if invert:
            return convert_to_epsinv(ODNF)
        return ODNF

    if target_language == circuits.Languages.MODS:
        ODNF = convert_ddnnf_to_odnf(circuit)
        MODS = convert_odnf_to_mods(ODNF)
        assert circuits.Languages.MODS in MODS.languages, "Should be MODS"
        if invert:
            return convert_to_epsinv(MODS)
        return MODS

    if target_language == circuits.Languages.OBDD:
        OBDD = convert_nnf_to_bdd(circuit)
        assert circuits.Languages.OBDD in OBDD.languages, "Should be OBDD, not " + str(OBDD.languages)
        return OBDD
