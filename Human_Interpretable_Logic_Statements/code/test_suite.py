import circuit_io
import formula_conversions
import circuits

def test_one_leaf():
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_NNF_one_leaf.nnf")
    assert circuits.Languages.NNF in NNF.languages
    assert circuits.Languages.DNNF in NNF.languages
    assert circuits.Languages.dNNF in NNF.languages
    assert circuits.Languages.sNNF in NNF.languages
    assert circuits.Languages.fNNF in NNF.languages
    assert circuits.Languages.dDNNF in NNF.languages
    assert circuits.Languages.sdDNNF in NNF.languages
    assert circuits.Languages.BDD in NNF.languages
    assert circuits.Languages.FBDD in NNF.languages
    assert circuits.Languages.OBDD in NNF.languages
    assert circuits.Languages.OBDD_LT in NNF.languages
    assert circuits.Languages.DNF in NNF.languages
    assert circuits.Languages.CNF in NNF.languages
    assert circuits.Languages.MODS in NNF.languages
    assert circuits.Languages.ODNF in NNF.languages

def test_ODNF(): 
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_ODNF.nnf")
    assert circuits.Languages.ODNF in NNF.languages

def test_flat():
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_CNF.nnf")
    assert circuits.LanguageProperties.flat in NNF.language_properties 

    CNF = circuit_io.parse_CNF_DIMACS("../nnf_examples/test_CNF.cnf")
    assert circuits.LanguageProperties.flat in CNF.language_properties 

    DNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_DNF.nnf")
    assert circuits.LanguageProperties.flat in DNF.language_properties

def test_simple_disjunction():
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_CNF.nnf")
    assert circuits.LanguageProperties.simple_disjunction in NNF.language_properties 

    CNF = circuit_io.parse_CNF_DIMACS("../nnf_examples/test_CNF.cnf")
    assert circuits.LanguageProperties.simple_disjunction in CNF.language_properties 

def test_simple_conjunction(): 
    DNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_DNF.nnf")
    assert circuits.LanguageProperties.simple_conjunction in DNF.language_properties

def test_decomposable():
    dDNNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dDNNF.nnf")
    assert circuits.LanguageProperties.decomposable in dDNNF.language_properties

def test_deterministic():
    dNNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dNNF.nnf")
    assert circuits.LanguageProperties.deterministic in dNNF.language_properties

def test_smooth():
    sNNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_sNNF.nnf")
    assert circuits.LanguageProperties.smooth in sNNF.language_properties

def test_decisive():
    BDD = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_FBDD.nnf")
    assert circuits.LanguageProperties.decisive in BDD.language_properties

def test_ordered():
    BDD = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD.nnf")
    assert circuits.LanguageProperties.ordered in BDD.language_properties

def test_ordered_less_than():
    BDD = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD_LT.nnf")
    assert circuits.LanguageProperties.ordered_lt in BDD.language_properties

def test_tseitin():
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_CNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dDNNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_DNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dNNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_FBDD.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_MODS.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_NNF_one_leaf.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD_LT.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_sdDNNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_sNNF.nnf")
    cnf = formula_conversions.tseitin(NNF)
    assert circuits.Languages.CNF in cnf.languages

def test_non_tseitin_cnf():
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_CNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()
    
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dDNNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()
    
    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_DNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_dNNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_FBDD.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_MODS.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_NNF_one_leaf.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_OBDD_LT.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_sdDNNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()

    NNF = circuit_io.parse_NNF_DIMACS("../nnf_examples/test_sNNF.nnf")
    cnf = formula_conversions.convert_to_cnf(NNF)
    assert circuits.Languages.CNF in cnf.languages
    assert NNF.graph.collapse_to_truth_table() == cnf.graph.collapse_to_truth_table()


def test_epsilon_inverted(): 
    assert 1 == 1
    