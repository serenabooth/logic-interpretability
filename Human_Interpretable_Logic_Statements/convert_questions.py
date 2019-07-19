import argparse
import sys
sys.path.append('../Human_Interpretable_Logic_Statements/code')
import circuits
import circuit_io

predicate_map = {}

def read_write_questions(predicates_fname, questions_input, questions_reformatted):
    """
    CAUTION: formatting of input and output precisely defined. 
    """
    with open(predicates_fname) as f:
        line = f.readline()
        while line:
            int_start = line.find(': ') + len(": ")
            int_end = line.find(' - ')
            line_end = line.find("\n")

            idx = int(line[int_start:int_end])
            predicate = line[int_end + len(" - "):line_end]
            predicate_map[predicate] = idx
            line = f.readline()

    print (predicate_map)

    fd = open(questions_reformatted, "w+")
    with open(questions_input) as f:
        line = f.readline()
        while line:
            if "When do you" in line: 
                line = f.readline()
            elif "\n" == line: 
                line = f.readline()
            elif "Time to answer" in line: 
                line = f.readline()                
            else: 
                line = line.replace("not ", " -")

                action_start = line.find("answer_question I do ") + len("answer_question I do ")
                line = line[action_start:]
                line = line.replace('()=None when ("', ": ((")
                line = line.replace("()=None when ('", ": ((")

                kc_end = max(line.find('"'), line.find("'"))
                line = line[:kc_end]
                line = line.replace('---or---', ') or (')
                line = line + "))"

                for predicate, val in predicate_map.items():
                    line = line.replace(predicate, str(val))
                fd.write(line + "\n")
                line = f.readline()

def convert_natural_language_to_DIMACS(input_fname, output_dir):
    with open(input_fname) as f: 
        for line in f: 
            if line == "\n" or line == None:
                continue
            else:
                dnf_expression = line[line.find(": ")+2:-1]
                print (dnf_expression)
                circuit = circuit_io.parse_DNF_natural_language(dnf_expression)
                output_name = line[:line.find(":")]
                circuit_io.write_NNF_DIMACS(circuit, output_dir + str(output_name) + ".nnf")
                # circuit.graph.graph_logic_formula()

def main():    
    parser = argparse.ArgumentParser(description='Which files to convert.')
    parser.add_argument("-d", "--domain", default="chopsticks", required=False, help="domain")

    args = vars(parser.parse_args())

    if args['domain'] == 'chopsticks':
        read_write_questions("domains/chopsticks_examples/good_agent/predicates.txt",
                             "domains/chopsticks_examples/good_agent/questions.txt",
                             "domains/chopsticks_examples/good_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/chopsticks_examples/good_agent/questions-formatted.txt", 
                                           "domains/chopsticks_examples/good_agent/")
        read_write_questions("domains/chopsticks_examples/bad_agent/predicates.txt",
                             "domains/chopsticks_examples/bad_agent/questions.txt",
                             "domains/chopsticks_examples/bad_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/chopsticks_examples/bad_agent/questions-formatted.txt", 
                                           "domains/chopsticks_examples/bad_agent/")
    elif args['domain'] == 'highway':
        read_write_questions("domains/highway_examples/good_agent/predicates.txt",
                             "domains/highway_examples/good_agent/questions.txt",
                             "domains/highway_examples/good_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/highway_examples/good_agent/questions-formatted.txt",
                             "domains/highway_examples/good_agent/")
        read_write_questions("domains/highway_examples/bad_agent/predicates.txt",
                             "domains/highway_examples/bad_agent/questions.txt",
                             "domains/highway_examples/bad_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/highway_examples/bad_agent/questions-formatted.txt",
                             "domains/highway_examples/bad_agent/")
    elif args['domain'] == 'emergency':
        read_write_questions("domains/emergency_triage_examples/good_agent/predicates.txt",
                             "domains/emergency_triage_examples/good_agent/questions.txt",
                             "domains/emergency_triage_examples/good_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/emergency_triage_examples/good_agent/questions-formatted.txt",
                             "domains/emergency_triage_examples/good_agent/")
        read_write_questions("domains/emergency_triage_examples/bad_agent/predicates.txt",
                             "domains/emergency_triage_examples/bad_agent/questions.txt",
                             "domains/emergency_triage_examples/bad_agent/questions-formatted.txt")
        convert_natural_language_to_DIMACS("domains/emergency_triage_examples/bad_agent/questions-formatted.txt",
                             "domains/emergency_triage_examples/bad_agent/")
if __name__ == "__main__":
    main()
