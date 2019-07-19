from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import sys
import itertools
import pprint
import argparse

import circuit_io
import formula_conversions

EPSILON = 0.5

class Languages(Enum):
    """
    Definitions in:
        Darwiche, A., & Marquis, P. (2002). A knowledge compilation map.
        Journal of Artificial Intelligence Research, 17, 229-264.

    ODNF - intersection of DNF and d-NNF
    """
    NNF = 1         # negation normal form
    DNNF = 2        # decomposable negation normal form
    dNNF = 3        # deterministic negation normal form
    sNNF = 4        # smooth negation normal form
    fNNF = 5        # flat negation normal form
    dDNNF = 6       # deterministic decomposable negation normal form
    sdDNNF = 7      # smooth deterministic decomposable negation normal form
    BDD = 8         # binary decision diagram
    FBDD = 9        # free binary decision diagram
    OBDD = 10       # ordered binary decision diagram
    OBDD_LT = 11    # ordered binary decision diagram, <
    DNF = 12        # disjunctive normal form
    CNF = 13        # conjunctive normal form
    PI = 14         # prime implicates
    IP = 15         # prime implicants
    MODS = 16       # models
    ODNF = 17       # orthoganol disjunctive normal form


class LanguageProperties(Enum):
    """
    Definitions in:
        Darwiche, A., & Marquis, P. (2002). A knowledge compilation map.
        Journal of Artificial Intelligence Research, 17, 229-264.
    """
    nnf_form = 1
    flat = 2
    simple_disjunction = 3
    simple_conjunction = 4
    decomposable = 5
    deterministic = 6
    smooth = 7
    decisive = 8
    ordered = 9
    ordered_lt = 10
    eps_inverted = 11


class DiGraph(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

    def find_dag_root(self):
        """
        Find the root of the DAG

        Parameters
        ----------
        self : DiGraph
            An instantiated DiGraph

        Returns
        -------
        int : Index of root node in graph, or -1 if none exists
        """
        for node in self.nodes():
            if self.in_degree(node) == 0:
                return node
        else:
            return -1

    def find_all_leaves(self):
        """
        Find the leaves of the DAG

        Parameters
        ----------
        self : DiGraph
            Instantiated DiGraph

        Returns
        -------
        List : List of ints, each int is an index of a leaf node in NNF graph
        """
        return [x for x in self.nodes() if self.out_degree(x)==0]

    def get_variables(self, clause_idx):
        """
        Given a node in the graph, get its composing variables

        Recursively gather variables of all child nodes (and their child nodes).
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        graph : DiGraph()
            DiGraph representing a logical statement
        clause_idx : int
            An index pointing to a node in a directed acyclic graph

        Returns
        -------
        child_variables : set
            A set of all variables composing the given clause
        """
        child_variables = set()

        # aux variables are added during translations between languages;
        # they are designated with an "aux_var" field to indicate this
        if 'aux_var' in self.node[clause_idx] and "-" in self.node[clause_idx]['value']:
            child_variables.add(self.node[clause_idx]['value'][1:])
        elif 'aux_var' in self.node[clause_idx] and "-" not in self.node[clause_idx]['value']:
                child_variables.add(self.node[clause_idx]['value'][:])
        elif self.node[clause_idx]['type'] == "LEAF" and isinstance(self.node[clause_idx]['value'], int):
            child_variables.add(abs(self.node[clause_idx]['value']))
        else:
            clause_children = self.neighbors(clause_idx)
            for child in clause_children:
                child_variables = child_variables.union(self.get_variables(child))
        return child_variables

    def collapse_graph_to_formula_promela_syntax(self, clause_idx):
        formula = ""
        # node is a leaf with integer value
        if self.node[clause_idx]['type'] == "LEAF" and isinstance(self.node[clause_idx]['value'], int):
            # leaves with value < 0 are negated
            if self.node[clause_idx]['value'] < 0:
                formula += "~" + str(abs(self.node[clause_idx]['value'])) + " "
            else:
                formula += " " + str(self.node[clause_idx]['value']) + " "
        # node is a leaf with tseitin-assigned string value
        elif self.node[clause_idx]['type'] == "LEAF" and not isinstance(self.node[clause_idx]['value'], int):
            # leaves starting with "-" are negated
            if self.node[clause_idx]['value'][0] == "-":
                formula += "~" + self.node[clause_idx]['value'][1:] + " "
            else:
                formula += " " + self.node[clause_idx]['value'] + " "
        elif len(list(self.neighbors(clause_idx))) > 0:
            # suppose clause is an AND, and children are nodes 1, 2, 3
            clause_children = list(self.neighbors(clause_idx))
            if self.node[clause_idx]['value'] == "nor" and len(clause_children) > 1:
                formula += "~"
            if self.node[clause_idx]['value'] == "nand" and len(clause_children) > 1:
                formula += "~"
            formula += "("
            child_clause_id = 0
            # for example, add " 1  and " + " 2  and "
            while child_clause_id < len(clause_children) - 1:
                formula += self.collapse_graph_to_formula_promela_syntax(clause_children[child_clause_id])
                if self.node[clause_idx]['value'] == "nand":
                    formula += " & "
                if self.node[clause_idx]['value'] == "and":
                    formula += " & "
                if self.node[clause_idx]['value'] == "nor":
                    formula += " | "
                if self.node[clause_idx]['value'] == "or":
                    formula += " | "
                child_clause_id += 1
            # for given example, add " 3 "
            formula += self.collapse_graph_to_formula_promela_syntax(clause_children[child_clause_id])
            formula += ")"
        return formula

    def collapse_graph_to_formula(self, clause_idx, natural_language=False):
        """
        Take a DAG, from some arbitrary node, and collapse
        it to the natural language formula. This formula can
        be evaluated by replacing the variables with True/False
        and running the command eval().

        Parameters
        ----------
        self : DiGraph
            DiGraph representing a logical statement
        clause_idx : int
            An integer which is an index to a node in the graph

        Returns
        -------
        formula : string
            A natural language form of the collapsed graph
        """
        formula = ""
        # node is a leaf with integer value
        if self.node[clause_idx]['type'] == "LEAF" and isinstance(self.node[clause_idx]['value'], int):
            # leaves with value < 0 are negated
            if self.node[clause_idx]['value'] < 0:
                formula += "not( " + str(abs(self.node[clause_idx]['value'])) + " )"
            else:
                formula += " " + str(self.node[clause_idx]['value']) + " "
        # node is a leaf with tseitin-assigned string value
        elif self.node[clause_idx]['type'] == "LEAF" and not isinstance(self.node[clause_idx]['value'], int):
            # leaves starting with "-" are negated
            if self.node[clause_idx]['value'][0] == "-":
                formula += "not( " + self.node[clause_idx]['value'][1:] + " )"
            else:
                formula += " " + self.node[clause_idx]['value'] + " "
        elif len(list(self.neighbors(clause_idx))) > 0:
            # suppose clause is an AND, and children are nodes 1, 2, 3
            clause_children = list(self.neighbors(clause_idx))
            if self.node[clause_idx]['value'] == "nor" and len(clause_children) > 1 and natural_language:
                formula += "neither"
            if self.node[clause_idx]['value'] == "nand" and len(clause_children) > 2 and natural_language:
                formula += "not all of "
            if self.node[clause_idx]['value'] == "nand" and len(clause_children) == 2 and natural_language:
                formula += "not both of "


            formula += "("
            if (self.node[clause_idx]['value'] == "nand" or self.node[clause_idx]['value'] == "nor") and \
                len(clause_children) == 1:
                formula += "not("
            child_clause_id = 0
            # for example, add " 1  and " + " 2  and "
            while child_clause_id < len(clause_children) - 1:
                formula += self.collapse_graph_to_formula(clause_children[child_clause_id], natural_language)
                if self.node[clause_idx]['value'] == "nand" and natural_language:
                    formula += " and "
                else:
                    formula += " " + self.node[clause_idx]['value'] + " "
                child_clause_id += 1
            # for given example, add " 3 "
            formula += self.collapse_graph_to_formula(clause_children[child_clause_id], natural_language)
            if (self.node[clause_idx]['value'] == "nand" or self.node[clause_idx]['value'] == "nor") and \
                len(clause_children) == 1:
                formula += ")"
            formula += ")"
        return formula

    def distance_node_to_nearest_leaf(self, clause_idx):
        i = 0
        if self.node[node]['type'] == "LEAF":
            return i
        elif self.node[node]['type'] == "LOGIC_GATE":
            i += 1
            list_of_distances = []
            for child in self.neighbors(node):
                list_of_distances.append(self.distance_node_to_nearest_leaf(child))
            return i + min(list_of_distances)

    def size_of_subgraph(self, clause_idx):
        """
        Return the size of the subgraph (number of nodes)

        Parameters
        ----------
        self : DiGraph
            DiGraph representing a circuit
        clause_idx : int
            Represent node index in graph

        Returns
        -------
        int : number of node descendants from clause_idx
        """
        return len(list(nx.descendants(self, clause_idx)))

    def _collapse_graph_to_bullet_list_helper(self, clause_idx, recursion_level, negate=None):
        """
        Helper: Take a DAG, from some arbitrary node, and collapse
        it to an HTML bulletted list. This formula can
        be evaluated by replacing the variables with True/False
        and running the command eval().

        Parameters
        ----------
        self : DiGraph
            DiGraph representing a logical statement
        clause_idx : int
            An integer which is an index to a node in the graph

        Returns
        -------
        html_formula : string
            A natural language form of the collapsed graph
        """
        if recursion_level > 1:
            recursion_level = 1
        ul_index = ['<ul style="list-style-type:circle;">', '<ul style="list-style-type:square;">']
        html_formula = ""
        if self.nodes[clause_idx]['type'] == "LEAF":
            if negate:
                html_formula += "<li> -" + str(self.nodes[clause_idx]['value']) + "</li>\n"
            else:
                html_formula += "<li>" + str(self.nodes[clause_idx]['value']) + "</li>\n"
        elif self.nodes[clause_idx]['type'] == "LOGIC_GATE":
            negate = None

            children = list(self.neighbors(clause_idx))
            children.sort(key = self.size_of_subgraph)

            if len(children) == 1:
                if self.nodes[clause_idx]['value'] == "or":
                    html_formula += ""
                elif self.nodes[clause_idx]['value'] == "nor":
                    negate = True
                elif self.nodes[clause_idx]['value'] == "and":
                    html_formula += ""
                elif self.nodes[clause_idx]['value'] == "nand":
                    negate = True
                for child in children:
                    html_formula += self._collapse_graph_to_bullet_list_helper(child, recursion_level + 1, negate)
                html_formula += "</li>\n"
            elif len(children) == 2:
                if self.nodes[clause_idx]['value'] == "or":
                    html_formula += "<li> One or both of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "nor":
                    html_formula += "<li> Neither of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "and":
                    html_formula += "<li> Both of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "nand":
                    html_formula += "<li> Not both of:\n" + ul_index[recursion_level] + "\n"
                for child in children:
                    html_formula += self._collapse_graph_to_bullet_list_helper(child, recursion_level + 1)
                html_formula += "</ul>\n</li>\n"
            else:
                if self.nodes[clause_idx]['value'] == "or":
                    html_formula += "<li> One or more of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "nor":
                    html_formula += "<li> None of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "and":
                    html_formula += "<li> All of:\n" + ul_index[recursion_level] + "\n"
                elif self.nodes[clause_idx]['value'] == "nand":
                    html_formula += "<li> Not all of:" + ul_index[recursion_level] + "\n"
                for child in children:
                    html_formula += self._collapse_graph_to_bullet_list_helper(child, recursion_level + 1)
                html_formula += "</ul>\n</li>\n"

        return html_formula

    def collapse_graph_to_bullet_list(self):
        """
        Collapse graph to HTML list

        Parameters:
        self : DiGraph
            Representing a circuit

        Returns:
        formula : string

        """
        formula = '<ul style="list-style-type:disc;">\n'
        root = self.find_dag_root()
        formula += self._collapse_graph_to_bullet_list_helper(root, 0)
        formula += "</ul>"

        return formula

    def graph_logic_formula(self):
        """
        Visualize a graph.

        Graph should consist of a directed acyclic graph,
        representing a logic statement.

        Parameters
        ----------
        self : DiGraph
            A graph consisting of a logic statement.

        OPTIONAL label : string
            A title for the graph

        Returns
        -------
        True

        """
        pos = graphviz_layout(self, prog="dot")
        node_attrs = nx.get_node_attributes(self, 'value')

        custom_node_attrs = {}
        for node, attr in node_attrs.items():
            custom_node_attrs[node] = str(node) + "." + str(attr)

        nx.draw(self,
            labels=custom_node_attrs,
            with_labels=True,
            node_size=14000,
            node_color="yellow",
            alpha=1,
            width=3,
            font_size=32,
            pos=pos)
        plt.show()

    def collapse_to_truth_table(self, clause_idx = None):
        """
        Print the truth table for a graph

        Parameters
        ----------
        self : NNF

        Returns
        -------
        truth_table : list of lists
            [['1', '2', '3', 'Outcome'], [' True ', ' False ', ' True ', False]]
            Notice that the first element in the list is a header

        """
        if not clause_idx:
            root = self.find_dag_root()
        else:
            root = clause_idx
        all_variables = self.get_variables(root)
        computed_formula = self.collapse_graph_to_formula(root)

        # generate a truth table for the variables
        table = list(itertools.product([" False ", " True "],
                            repeat=len(all_variables)))


        truth_table = [list(all_variables) + ["Outcome"]]
        idx = 0
        for truth_assignment in table:
            formula_evals_to = None
            # create a new copy of the formula (not just a pointer)
            tmp_formula = str(computed_formula)
            for i in range(0, len(all_variables)):
                tmp_formula = tmp_formula.replace(
                                    " " + str(list(all_variables)[i]) + " ",
                                    truth_assignment[i])
            truth_table.append(list(truth_assignment) + [eval(tmp_formula)])


        return truth_table

    def collapse_to_satisfied(self):
        """
        Print the satisfying assignment for a graph

        Parameters
        ----------
        self : NNF

        Returns
        -------
        satisfying_assignments : list of lists
            [['1', '2', '3', 'Outcome'], [' True ', ' False ', ' True ', True]]
            Notice that the first element in the list is a header

        """
        root = self.find_dag_root()
        all_variables = self.get_variables(root)
        computed_formula = self.collapse_graph_to_formula(root)

        # generate a truth table for the variables
        table = list(itertools.product([" False ", " True "],
                            repeat=len(all_variables)))


        satisfying_assignments = [list(all_variables) + ["Outcome"]]
        idx = 0
        for truth_assignment in table:
            formula_evals_to = None
            # create a new copy of the formula (not just a pointer)
            tmp_formula = str(computed_formula)
            for i in range(0, len(all_variables)):
                tmp_formula = tmp_formula.replace(
                                    " " + str(list(all_variables)[i]) + " ",
                                    truth_assignment[i])
            if eval(tmp_formula):
                satisfying_assignments.append(list(truth_assignment) + [eval(tmp_formula)])

        return satisfying_assignments

class Circuit:
    def __init__(self,
                    clauses,
                    variables,
                    num_edges,
                    num_nodes,
                    graph_representation):
        self.clauses = clauses
        self.variables = variables
        self.graph = graph_representation
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.update_language_properties()
        self.update_language()


    def update_language_properties(self):
        self.language_properties = []
        # first, determine which properties are satisfied
        if self.is_nnf():
            self.language_properties.append(LanguageProperties.nnf_form)
        if self.is_flat():
            self.language_properties.append(LanguageProperties.flat)
        if self.is_simple_disjunction():
            self.language_properties.append(LanguageProperties.simple_disjunction)
        if self.is_simple_conjunction():
            self.language_properties.append(LanguageProperties.simple_conjunction)
        if self.is_decomposable():
            self.language_properties.append(LanguageProperties.decomposable)
        if self.is_deterministic():
            self.language_properties.append(LanguageProperties.deterministic)
        if self.is_smooth():
            self.language_properties.append(LanguageProperties.smooth)
        if self.is_decisive():
            self.language_properties.append(LanguageProperties.decisive)
        if self.is_ordered():
            self.language_properties.append(LanguageProperties.ordered)
        if self.is_ordered_less_than():
            self.language_properties.append(LanguageProperties.ordered_lt)
        if self.is_epsilon_inverted():
            self.language_properties.append(LanguageProperties.eps_inverted)


    def update_language(self):
        self.languages = []
        # Short-circuit if we aren't even NNF
        if LanguageProperties.nnf_form not in self.language_properties:
            return
        # determine which languages are satisfied (based on properties)
        if LanguageProperties.nnf_form in self.language_properties:
            self.languages.append(Languages.NNF)
        if LanguageProperties.flat in self.language_properties:
            self.languages.append(Languages.fNNF)
        if LanguageProperties.flat in self.language_properties and \
                    LanguageProperties.simple_disjunction in self.language_properties:
            self.languages.append(Languages.CNF)
        if LanguageProperties.flat in self.language_properties and \
                    LanguageProperties.simple_conjunction in self.language_properties:
            self.languages.append(Languages.DNF)
        if LanguageProperties.decomposable in self.language_properties:
            self.languages.append(Languages.DNNF)
        if LanguageProperties.deterministic in self.language_properties:
            self.languages.append(Languages.dNNF)
        if LanguageProperties.smooth in self.language_properties:
            self.languages.append(Languages.sNNF)
        if LanguageProperties.deterministic in self.language_properties and \
                    LanguageProperties.decomposable in self.language_properties:
            self.languages.append(Languages.dDNNF)
        if LanguageProperties.smooth in self.language_properties and \
                    LanguageProperties.deterministic in self.language_properties and \
                    LanguageProperties.decomposable in self.language_properties:
            self.languages.append(Languages.sdDNNF)
        if LanguageProperties.decisive in self.language_properties:
            self.languages.append(Languages.BDD)
        if LanguageProperties.decisive in self.language_properties and \
            LanguageProperties.decomposable in self.language_properties:
            self.languages.append(Languages.FBDD)
        if LanguageProperties.decisive in self.language_properties and \
            LanguageProperties.decomposable in self.language_properties and \
            LanguageProperties.ordered in self.language_properties:
            self.languages.append(Languages.OBDD)
        if LanguageProperties.decisive in self.language_properties and \
            LanguageProperties.decomposable in self.language_properties and \
            LanguageProperties.ordered_lt in self.language_properties:
            self.languages.append(Languages.OBDD_LT)
        if LanguageProperties.flat in self.language_properties and \
                    LanguageProperties.simple_conjunction in self.language_properties and \
                    LanguageProperties.deterministic in self.language_properties and \
                    LanguageProperties.smooth in self.language_properties:
            self.languages.append(Languages.MODS)
        if LanguageProperties.flat in self.language_properties and \
                    LanguageProperties.simple_conjunction in self.language_properties and \
                    LanguageProperties.deterministic in self.language_properties:
            self.languages.append(Languages.ODNF)
        # leaving out IP + PI

    def simplify_cnf(self):
        if not self.is_flat():
            return self
        if not self.is_simple_disjunction():
            return self

        root = self.graph.find_dag_root()

        truth_tables = []
        for child in self.graph.neighbors(root):
            child_table = self.graph.collapse_to_truth_table(child)
            in_table = False
            for table in truth_tables:
                if child_table == table:
                    self.graph.remove_node(child)
                    in_table = True
            if not in_table:
                truth_tables.append(child_table)

        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0 and self.graph.in_degree(node) == 0:
                graph.remove_node(node)

        # self.graph.graph_logic_formula()
        self.update_language_properties()
        self.update_language()

        return self


    def is_nnf(self):
        """
        Determine whether a statement is NNF.

        NNF definition: each leaf node is labeled True, False, X, or -X;
        each internal node is labeled with "and" or "or". Negation is only
        applied to literals.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph representation

        Returns
        -------
        True iff statement meets NNF criteria

        """
        for node in self.clauses:
            if self.graph.node[node]['value'] in ["nand", "nor"]:
                return False
        return True


    def is_flat(self):
        """
        Determine whether a logical statement is flat.

        Flat definition: the height of Circuit is at most 2.
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets flat criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        if nx.dag_longest_path_length(self.graph) <= 2:
            return True
        return False


    def is_simple_disjunction(self):
        """
        Determine whether a logical statement meets simple disjunction.

        Simple disjunction definition: every disjunction is a clause,
        where literals do not share variables.
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets simple disjunction criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        # look through all or clauses in logic sentence
        or_terms = [node for node in self.clauses if self.graph.node[node]['value'] == "or"]

        for clause_idx in or_terms:
            clause_children = self.graph.neighbors(clause_idx)
            # check if each child consists of a leaf or logic gate
            for child in clause_children:
                if self.graph.node[child]['type'] == "LOGIC_GATE":
                    return False
        return True


    def is_simple_conjunction(self):
        """
        Determine whether a logical statement meets simple conjunction.

        Simple conjunction definition: every disjunction is a term,
        where literals do not share variables.
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets simple conjunction criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        # look through all and clauses in logic sentence
        and_clauses = [node for node in self.clauses if self.graph.node[node]['value'] == "and"]

        for clause_idx in and_clauses:
            clause_children = self.graph.neighbors(clause_idx)
            # check if each child consists of a leaf or logic gate
            for child in clause_children:
                if self.graph.node[child]['type'] == "LOGIC_GATE":
                    return False
        return True


    def is_decomposable(self):
        """
        Determine whether a logical statement is decomposable.

        Decomposable definition: conjuncts do not share variables.
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets decomposable criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        # look through all clauses in logic sentence
        and_clauses = [node for node in self.clauses if self.graph.node[node]['value'] == "and"]

        for clause_idx in and_clauses:
            all_variables = set()
            clause_children = self.graph.neighbors(clause_idx)
            # for each child of the "AND" clause, create a set of its variables
            for child in clause_children:
                child_variables = self.graph.get_variables(child)
                # if two children share variables, statement is not decomposable
                if len(all_variables.intersection(child_variables)) > 0:
                    return False
                else:
                    all_variables = all_variables.union(child_variables)
        return True


    def is_deterministic(self):
        """
        Determine whether a logical statement is deterministic.

        Deterministic definition: disjuncts are logically disjoint.
        Must have an instantiated directed acyclic graph representation.

        This function is computationally intensive.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets deterministic criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        # find all of the "OR" clauses
        or_terms = [node for node in self.clauses if self.graph.node[node]['value'] == "or"]

        for clause_idx in or_terms:
            # get all variables used in the term
            all_variables_in_OR_clause = set(self.graph.get_variables(clause_idx))
            formulas = []
            clause_children = self.graph.neighbors(clause_idx)
            # create a formula for each disjunct of the OR clause
            for clause in clause_children:
                computed_formula = self.graph.collapse_graph_to_formula(clause)
                formulas.append(computed_formula)
            # generate a truth table for the variables
            table = list(itertools.product([" False ", " True "],
                                repeat=len(all_variables_in_OR_clause)))
            for truth_assignment in table:
                formula_evals_to = None
                # create a new copy of the formulas (not just a pointer)
                tmp_formulas = list(formulas)
                for i in range(0, len(all_variables_in_OR_clause)):
                    for j in range(0, len(formulas)):
                        tmp_formulas[j] = tmp_formulas[j].replace(
                                        " " + str(list(all_variables_in_OR_clause)[i]) + " ",
                                        truth_assignment[i])

                # if any two disjuncts evaluate True, they are logically disjoint so return False
                formula_evals_to = False
                for i in range(0, len(tmp_formulas)):
                    if eval(tmp_formulas[i]) and formula_evals_to:
                        return False
        return True


    def is_smooth(self):
        """
        Determine whether a logical statement is smooth.

        Smooth definition: disjuncts mention the same set of variables.
        Must have an instantiated directed acyclic graph representation.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets smooth criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        or_terms = [node for node in self.clauses if self.graph.node[node]['value'] == "or"]

        for clause_idx in or_terms:
            all_variables = None
            clause_children = self.graph.neighbors(clause_idx)
            for child in clause_children:
                if all_variables:
                    tmp_variables = self.graph.get_variables(child)
                    # if two children don't share variables, statement is not smooth
                    if all_variables.intersection(tmp_variables) != all_variables:
                        return False
                else:
                    all_variables = self.graph.get_variables(child)
        return True


    def is_decisive(self):
        """
        Determine whether a logical statement is 'decisive'.

        Decisive definition: Every OR node satisfies the decision property
        Must have an instantiated directed acyclic graph representation,
        and the root of the DAG is an OR node.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets decisive criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        root = self.graph.find_dag_root()

        # is the root a leaf?
        if self.graph.node[root]['type'] == "LEAF":
            return True
        # is the root an OR node?
        if self.graph.node[root]['value'] != "or":
            return False
        # do all OR nodes adhere to the decision property?
        or_terms = [node for node in self.clauses if self.graph.node[node]['value'] == "or"]
        for clause_idx in or_terms:
            is_decision_node, _ = self.is_decision_node(clause_idx)
            if not is_decision_node:
                return False
        return True


    def is_ordered(self):
        """
        Determine whether a logical statement is 'ordered'.

        Ordered definition: decision variables appear in the same order on any path in the Circuit

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets ordered criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False
        # must be decisive
        if not self.is_decisive():
            return False

        root = self.graph.find_dag_root()
        assert(root > -1), "No root found"
        leaves = self.graph.find_all_leaves()

        all_decision_variable_paths = self.all_decision_paths_root_to_leaves(root, leaves)

        return self.is_consistently_ordered(all_decision_variable_paths)


    def is_ordered_less_than(self):
        """
        Determine whether a logical statement is 'ordered' (alphanumerically).

        Ordered (less than) definition: decision variables appear in alphanumeric order on any path in the Circuit.

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets ordered_less_than criteria.
        """
        # must be NNF
        if not self.is_nnf():
            return False

        if not self.is_decisive():
            return False

        root = self.graph.find_dag_root()
        assert(root > -1), "No root found"
        leaves = self.graph.find_all_leaves()

        all_decision_variable_paths = self.all_decision_paths_root_to_leaves(root, leaves)

        return self.is_alphanumerically_ordered(all_decision_variable_paths)


    def all_decision_paths_root_to_leaves(self, root, leaves):
        """
        Starting at the root, find all decision nodes en route to each leaf

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation
        root : int
            Index of top of DAG node
        leaves : list of ints
            List of indices of DAG leaves

        Returns
        -------
        all_decision_variable_paths : list of list of ints
            A list of list of integers, representing pathways from root to leaves.
            For example: [[1, 2, 3], [1, 3]] would visit decision nodes 1, 2, 3 on
            one pathway, and nodes 1 and 3 on another

        """
         # find all paths from root -> each leaf
        all_paths = []
        for leaf in leaves:
            for path in nx.all_simple_paths(self.graph, root, leaf):
                all_paths.append(path)

        # find decision variables visited in each path from root -> leaf
        all_decision_variable_paths = []
        for path in all_paths:
            decision_variables_visited = []
            for node in path:
                is_decision_var, decision_vars = self.is_decision_node(node)
                if is_decision_var:
                    decision_variables_visited.append(decision_vars)
            all_decision_variable_paths.append(decision_variables_visited)

        return all_decision_variable_paths


    def is_consistently_ordered(self, all_decision_variable_paths):
        """
        Given list of all pathways from root of DAG to each leaf,
        and given an ordering pattern, determine whether the
        pathways adhere to the ordering.

        Parameters
        ----------
        all_decision_variable_paths : list of list of ints
            A list of list of integers, representing pathways from root to leaves.
            For example: [[1, 2, 3], [1, 3]] would visit decision nodes 1, 2, 3 on
            one pathway, and nodes 1 and 3 on another

        Returns
        -------
        True iff all the given decision variable pathways are consistently ordered
        """
        all_ordering = []

        for path in all_decision_variable_paths:
            sub_ordering = []

            for variables in path:
                if len(variables) == 1:
                    if abs(variables[0]) in all_ordering:
                        sub_ordering.append(all_ordering.index(abs(variables[0])))
                    else:
                        all_ordering.append(abs(variables[0]))
                else:
                    reordered_variables = sorted(variables)
                    # if there are multiple decision variables accessible at
                    # a given node, order them alphanumerically
                    for variable in reordered_variables:
                        if abs(variable) in all_ordering:
                            sub_ordering.append(all_ordering.index(abs(variable)))
                        else:
                            all_ordering.append(abs(variable))

            highest_index = -1
            for idx in sub_ordering:
                if idx < highest_index:
                    return False
                highest_index = idx

        return True

    def is_alphanumerically_ordered(self, all_decision_variable_paths):
        """
        Given list of all pathways from root of DAG to each leaf,
        and given an ordering pattern, determine whether the
        pathways adhere to the ordering.

        Parameters
        ----------
        all_decision_variable_paths : list
            A list of list of integers, representing pathways from root to leaves.
            For example: [[1, 2, 3], [1, 3]] would visit decision nodes 1, 2, 3 on
            one pathway, and nodes 1 and 3 on another

        Returns
        -------
        True iff all the given decision variable pathways are alphanumerically ordered
        """

        for path in all_decision_variable_paths:
            last_node = -1

            for variables in path:
                if len(variables) == 1:
                    if abs(variables[0]) < last_node:
                        return False
                    else:
                        last_node = abs(variables[0])

                else:
                    reordered_variables = sorted(variables)
                    # if there are multiple decision variables accessible at
                    # a given node, order them alphanumerically
                    for variable in reordered_variables:
                        if abs(variable) < last_node:
                            return False
                        else:
                            last_node = abs(variable)
        return True


    def _is_epsilon_inverted(self, nid, node):
        # Leaves always satisfy the property
        if node['type'] == 'LEAF':
            return True
        # Gather the negative children
        negations = [n for n in self.graph.neighbors(nid) if self.is_negation_node(self.graph.nodes()[n])]
        # Return based on the ratio
        if len(list(self.graph.neighbors(nid))) == 0:
            return True
        else:
            return (len(negations) / len(list(self.graph.neighbors(nid)))) <= EPSILON


    def _negate_node(self, node):
        if node['type'] == 'LEAF':
            if isinstance(node['value'], int):
                node['value'] = -1 * node['value']
            else:
                assert(isinstance(node['value'], str))
                if node['value'] == "True":
                    node['value'] = "False"
                elif node['value'] == "False":
                    node['value'] = "True"
                elif '-' == node['value'][0]:
                    node['value'] = node['value'][1:]
                else:
                    node['value'] = '-'+node['value']
        else:
            assert node['type'] == 'LOGIC_GATE'
            node['value'] = {
                'or': 'nand',
                'nor': 'and',
                'and': 'nor',
                'nand': 'or'
            }[node['value']]


    def is_epsilon_inverted(self):
        """
        Determine whether a logical statement is epsilon inverted.

        Definition: children of OR and AND nodes have no more than an EPSILON chance
                    of being a NOT node. Note that this is strictly not an NNF
                    representation.

        Must have an instantiated directed acyclic graph representation.

        Should be quick to compute.

        Parameters
        ----------
        self : NF
            NF instantiation with graph_representation

        Returns
        -------
        True iff statement meets eps-INF criteria.
        """

        # Check all of the nodes
        for (nid, node) in self.graph.nodes().items():
            if not self._is_epsilon_inverted(nid, node):
                return False
        return True


    def is_negation_node(self, node):
        return (node['value'] in ['nand','nor']) or \
               (node['type'] == 'LEAF' and isinstance(node['value'], int) and (node['value'] < 0)) or \
               (node['type'] == 'LEAF' and isinstance(node['value'], str) and ('-' == node['value'][0]))


    def is_nnf(self):
        """
        Determine whether a logical statement is in negation normal form.

        Definition: Just a property that holds when negations are only at the leaves

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation

        Returns
        -------
        True iff statement meets NNF criteria.
        """
        for n in self.graph.nodes().values():
            if (n['type'] == 'LOGIC_GATE') and self.is_negation_node(n):
                return False
        return True


    def is_decision_node(self, clause_idx):
        """
        Given a pointer to a node in a DAG, determine whether
        this node satisfies the decision property.
        Namely:
            * the node itself is an OR node
            * it has children of the form (X and alpha) and (-X and beta)
            * alpha and beta may be null

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation
        clause_idx : int
            An integer which is an index to a node in the graph of the Circuit

        Returns
        -------
        Boolean
            True iff the node at clause_idx satisfies the decision property
        decision_var_value
            A list of the decision variable(s) associated with the node.
            Note this refers to the decision variables *by value*,
            NOT by index, as is typical
        """
        if self.graph.node[clause_idx]['value'] != "or":
            return (False, [])
        else:
            variables = []
            for child in self.graph.neighbors(clause_idx):
                variables_per_child = []
                if self.graph.node[child]['value'] == "and":
                    for grandchild in self.get_and_descendents(child):
                        if self.graph.node[grandchild]['type'] == "LEAF" and isinstance(self.graph.node[grandchild]['value'], int):
                            variables_per_child.append(int(self.graph.node[grandchild]['value']))
                elif self.graph.node[child]['type'] == "LEAF" and isinstance(self.graph.node[child]['value'], int):
                    variables_per_child.append(int(self.graph.node[child]['value']))
                elif self.graph.node[child]['value'] == "or":
                    return (False, [])
                variables.append(variables_per_child)

            if len(variables) < 2:
                return (False, [])

        for i in range(0, len(variables) - 1):
            decision_var_value = []
            for j in range(i+1, len(variables)):
                found_match = False
                for var_outer in variables[i]:
                    for var_inner in variables[j]:
                        if -1 * var_outer == var_inner:
                            found_match = True
                if not found_match:
                    return (False, decision_var_value)
        return (True, decision_var_value)


    def get_and_descendents(self, clause_idx):
        """
        Given a node:
            - if it's an AND, return a list of all of its descendants
            - if it's a LEAF or an OR, return a list of the given node

        Parameters
        ----------
        self : Circuit
            Circuit instantiation with graph_representation
        clause_idx : int
            An integer which is an index to a node in the graph of the Circuit

        Returns
        -------
        List of descendant nodes
        """
        list_of_child_nodes = []
        if self.graph.node[clause_idx]['value'] == "and":
            for child in self.graph.neighbors(clause_idx):
                list_of_child_nodes += self.get_and_descendents(child)
            return list_of_child_nodes
        else:
            return [clause_idx]


    def _is_mutually_inconsistent(self, clause_idx_1, clause_idx_2):
        """
        Given two pointers into DAG, determine if their composition is mutually inconsistent

        Parameters:
        ----------
        self : Circuit
            Must have instantiated DAG representation
        clause_idx_1 : int
            Corresponds to node in graph
        clause_idx_2 : int
            Corresponds to node in graph

        Returns:
        -------
        True iff every truth assignment is mutually inconsistent for the two terms
        """
        clause_1_vars = self.graph.get_variables(clause_idx_1)
        clause_2_vars = self.graph.get_variables(clause_idx_2)

        clause_1_formula = self.graph.collapse_graph_to_formula(clause_idx_1)
        clause_2_formula = self.graph.collapse_graph_to_formula(clause_idx_2)


        intersection = list(set(clause_1_vars) & set(clause_2_vars))
        union = list(set(clause_1_vars).union(set(clause_2_vars)))

        table = list(itertools.product([" False ", " True "],
                            repeat=len(union)))

        # NOT mutually exclusive
        if len(intersection) == 0:
            return False

        for truth_assignment in table:
            tmp_clause_1_evals_to = str(clause_1_formula)
            tmp_clause_2_evals_to = str(clause_2_formula)

            for i in range(0, len(union)):
                tmp_clause_1_evals_to = tmp_clause_1_evals_to.replace(" " + str(list(union)[i]) + " ",
                                    truth_assignment[i])
                tmp_clause_2_evals_to = tmp_clause_2_evals_to.replace(" " + str(list(union)[i]) + " ",
                                    truth_assignment[i])

            # both statements cannot be true (but both can be false)
            if eval(tmp_clause_1_evals_to) == eval(tmp_clause_2_evals_to) == True:
                return False

        return True


def main():
    parser = argparse.ArgumentParser(description='Determine which conversions to do.')
    parser.add_argument("-f", "--file", required=False, help="File name to convert")
    parser.add_argument("-p", "--paper", required=False, help="Generate images for paper")

    args = vars(parser.parse_args())

    if args["file"]:
        dnf = circuit_io.parse_NNF_DIMACS(args["file"])
        dnf.graph.graph_logic_formula()

        # used for dDNNF and sdDNNF conversions
        cnf = formula_conversions.convert_to_cnf(dnf)

        cnf = formula_conversions.simplify_cnf(cnf)
        cnf.graph.graph_logic_formula()

        dDNNF = formula_conversions.convert_nnf_to_dDNNF(cnf)
        dDNNF.graph.graph_logic_formula()

        e_inv_cnf = formula_conversions.convert_to_epsinv(cnf)
        e_inv_cnf.graph.graph_logic_formula()

        e_inv_dnf = formula_conversions.convert_to_epsinv(dnf)
        e_inv_dnf.graph.graph_logic_formula()

        e_inv_ddnnf = formula_conversions.convert_to_epsinv(dDNNF)
        e_inv_ddnnf.graph.graph_logic_formula()

        sdDNNF = formula_conversions.convert_nnf_to_sdDNNF(cnf)
        sdDNNF.graph.graph_logic_formula()

        ODNF = formula_conversions.convert_ddnnf_to_odnf(dnf)
        ODNF.graph.graph_logic_formula()

        MODS = formula_conversions.convert_odnf_to_mods(ODNF)
        MODS.graph.graph_logic_formula()

        formulas = circuit_io.write_file("tmp.txt")
        formulas.write("c\n")
        formulas.write("l DNF\n")
        formulas.write("f " + str(dnf.graph.collapse_graph_to_formula(dnf.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(dnf.num_edges) + "\n")
        formulas.write("p " + str(dnf.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l cnf\n")
        formulas.write("f " + str(cnf.graph.collapse_graph_to_formula(cnf.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(cnf.num_edges) + "\n")
        formulas.write("p " + str(cnf.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l dDNNF\n")
        formulas.write("f " + str(dDNNF.graph.collapse_graph_to_formula(dDNNF.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(dDNNF.num_edges) + "\n")
        formulas.write("p " + str(dDNNF.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l epsinv cnf\n")
        formulas.write("f " + str(e_inv_cnf.graph.collapse_graph_to_formula(e_inv_cnf.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(e_inv_cnf.num_edges) + "\n")
        formulas.write("p " + str(e_inv_cnf.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l epsinv dnf\n")
        formulas.write("f " + str(e_inv_dnf.graph.collapse_graph_to_formula(e_inv_dnf.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(e_inv_cnf.num_edges) + "\n")
        formulas.write("p " + str(e_inv_dnf.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l epsinv ddnnf\n")
        formulas.write("f " + str(e_inv_ddnnf.graph.collapse_graph_to_formula(e_inv_ddnnf.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(e_inv_ddnnf.num_edges) + "\n")
        formulas.write("p " + str(e_inv_ddnnf.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l sdDNNF\n")
        formulas.write("f " + str(sdDNNF.graph.collapse_graph_to_formula(sdDNNF.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(sdDNNF.num_edges) + "\n")
        formulas.write("p " + str(sdDNNF.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l ODNF\n")
        formulas.write("f " + str(ODNF.graph.collapse_graph_to_formula(ODNF.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(ODNF.num_edges) + "\n")
        formulas.write("p " + str(ODNF.languages) + "\n")

        formulas.write("c\n")
        formulas.write("l MODS\n")
        formulas.write("f " + str(MODS.graph.collapse_graph_to_formula(MODS.graph.find_dag_root())) + "\n")
        formulas.write("s " + str(MODS.num_edges) + "\n")
        formulas.write("p " + str(MODS.languages) + "\n")

        formulas.close()

if __name__ == "__main__":
    main()
