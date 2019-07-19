# -*- coding: UTF-8 -*-
from PIL import Image, ImageDraw, ImageFont
import csv
import random
import re
import argparse
import os
import sys
sys.path.append('../code')
import circuits
import circuit_io
import formula_conversions

action_natural_language_map = {"add_left_left" : "I add my left hand to my opponent's left hand",
                               "add_left_right" : "I add my left hand to my opponent's right hand",
                               "add_right_left" : "I add my right hand to my opponent's left hand",
                               "add_right_right" : "I add my right hand to my opponent's right hand",
                               "redistribute" : "I split my total",
                               "slow_down": "I slow down",
                               "increase_speed": "I speed up",
                               "merge_left": "I merge left",
                               "merge_right": "I merge right",
                               "do_nothing": "I do not take an action",
                               "immediate": "I triage as immediate",
                               "urgent": "I triage as urgent",
                               }

def generate_predicate_map(predicates_loc):
    predicate_map = {}
    with open(predicates_loc) as f:
        line = f.readline()
        while line:
            int_start = line.find(': ', 1) + len(": ")
            int_end = line.find(' - ')
            line_end = line.find("\n")

            idx = int(line[int_start:int_end])
            predicate = line[int_end + len(" - "):line_end]
            predicate_map[predicate] = idx
            line = f.readline()
    return predicate_map

def evaluate_predicates(state, directory, domain):
    def even_total(predicate_map):
        predicate_val = predicate_map["my total is even"]
        if (state[0][0] + state[0][1]) % 2 == 0:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def left_out(predicate_map):
        predicate_val = predicate_map["my left hand is out"]
        if (state[0][0] == 0):
            return int(predicate_val)
        return -1 * int(predicate_val)

    def right_out(predicate_map):
        predicate_val = predicate_map["my right hand is out"]
        if (state[0][1] == 0):
            return int(predicate_val)
        return -1 * int(predicate_val)

    def can_empty_opponent_left_lhs(predicate_map):
        predicate_val = predicate_map["my left hand outs my opponent's left hand"]
        if (state[0][0] + state[1][0]) % 5 == 0 and \
            state[0][0] != 0:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def can_empty_opponent_right_lhs(predicate_map):
        predicate_val = predicate_map["my right hand outs my opponent's left hand"]
        if (state[0][1] + state[1][0]) % 5 == 0 and \
            state[0][1] != 0:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def can_empty_opponent_left_rhs(predicate_map):
        predicate_val = predicate_map["my left hand outs my opponent's right hand"]
        if (state[0][0] + state[1][1]) % 5 == 0 and \
            state[0][0] != 0:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def can_empty_opponent_right_rhs(predicate_map):
        predicate_val = predicate_map["my right hand outs my opponent's right hand"]
        if (state[0][1] + state[1][1]) % 5 == 0 and \
            state[0][1] != 0:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def alert_able_to_move(predicate_map):
        predicate_val = predicate_map["patient is alert and able to move"]
        if (state[0]) == 1:
            return predicate_val
        return -1 * predicate_val

    def can_breathe(predicate_map):
        predicate_val = predicate_map["patient is breathing"]
        if (state[1]) == 1:
            return predicate_val
        return -1 * predicate_val

    def open_airway(predicate_map):
        predicate_val = predicate_map["patient is able to breathe after opening airway"]
        if (state[2]) == 1:
            return predicate_val
        return -1 * predicate_val

    def respiration_rate(predicate_map):
        predicate_val = predicate_map["respiration count is between 10 and 30"]
        if (state[3] >= 10) and (state[3] <= 30):
            return predicate_val
        return -1 * predicate_val

    def capillary_refill(predicate_map):
        predicate_val = predicate_map["capillary refill takes less than 2 seconds"]
        if (state[4] <= 2):
            return predicate_val
        return -1 * predicate_val

    def pulse_rate(predicate_map):
        predicate_val = predicate_map["pulse is between 40 and 70"]
        if (state[5] >= 40) and (state[5] <= 70):
            return predicate_val
        return -1 * predicate_val

    def car_left(predicate_map):
        predicate_val = predicate_map["a vehicle is to my left"]
        if state[0][0] == 1:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def car_right(predicate_map):
        predicate_val = predicate_map["a vehicle is to my right"]
        if state[2][0] == 1:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def car_ahead(predicate_map):
        predicate_val = predicate_map["a vehicle is in front of me"]
        if state[1][0] == 1:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def car_behind(predicate_map):
        predicate_val = predicate_map["a vehicle is behind me"]
        if state[1][2] == 1:
            return int(predicate_val)
        return -1 * int(predicate_val)

    def next_exit(predicate_map):
        predicate_val = predicate_map["the next exit is 42"]
        if state[3][0] == 42:
            return int(predicate_val)
        return -1 * int(predicate_val)

    predicate_map = generate_predicate_map(directory + "predicates.txt")
    predicates = []
    if domain == "chopsticks":
        predicates.append(even_total(predicate_map))
        predicates.append(left_out(predicate_map))
        predicates.append(right_out(predicate_map))
        predicates.append(can_empty_opponent_right_rhs(predicate_map))
        predicates.append(can_empty_opponent_left_rhs(predicate_map))
        predicates.append(can_empty_opponent_right_lhs(predicate_map))
        predicates.append(can_empty_opponent_left_lhs(predicate_map))
        # predicates.append(cannot_empty_opponent_state(predicate_map))
    if domain == "highway":
        predicates.append(car_left(predicate_map))
        predicates.append(car_right(predicate_map))
        predicates.append(car_ahead(predicate_map))
        predicates.append(car_behind(predicate_map))
        predicates.append(next_exit(predicate_map))
    if domain == "emergency":
        predicates.append(alert_able_to_move(predicate_map))
        predicates.append(can_breathe(predicate_map))
        predicates.append(open_airway(predicate_map))
        predicates.append(respiration_rate(predicate_map))
        predicates.append(capillary_refill(predicate_map))
        predicates.append(pulse_rate(predicate_map))
    return predicates

def possible_actions(predicate_values, questions_loc):
    actions = []
    with open(questions_loc) as f:
        line = f.readline()
        while line:
            action_end = line.find(': ', 1)
            line_end = line.find('\n')

            action = line[0:action_end]
            expression = line[action_end + len(": "):line_end]

            for predicate in predicate_values:
                if predicate < 0:
                    expression = expression.replace(str(predicate), "True")
                    expression = expression.replace(str(abs(predicate)), "False")
                elif predicate > 0:
                    expression = expression.replace(str(predicate), "True")

            expression = expression.replace("-", " not ")


            actions.append((action, eval(expression)))
            line = f.readline()
        f.close()

    return actions

def random_action(actions, truth_assignment):
    action_list = []
    for action in actions:
        if action[1] == truth_assignment:
            action_list.append(action[0])
    if len(action_list) == 0:
        return None
    return random.choice(action_list)

def generate_image_chopsticks(state):
    left = state[0][0]
    right = state[0][1]
    op_left = state[1][0]
    op_right = state[1][1]
    blank_image = Image.new('RGBA', (1600, 1600), 'white')

    my_left = Image.open("../assets/Chopsticks_PNG_formatted/hand_" + str(left) + "_left.png")
    my_right = Image.open("../assets/Chopsticks_PNG_formatted/hand_" + str(right) + "_right.png")
    opp_visual_right = Image.open("../assets/Chopsticks_PNG_formatted/hand_" + str(op_left) + "_left.png").transpose(Image.FLIP_TOP_BOTTOM)
    opp_visual_left = Image.open("../assets/Chopsticks_PNG_formatted/hand_" + str(op_right) + "_right.png").transpose(Image.FLIP_TOP_BOTTOM)

    my_left_label = Image.open("../assets/Chopsticks_PNG_formatted/my_left_label.png")
    my_right_label = Image.open("../assets/Chopsticks_PNG_formatted/my_right_label.png")
    opponent_left_label = Image.open("../assets/Chopsticks_PNG_formatted/opponent_left_label.png")
    opponent_right_label = Image.open("../assets/Chopsticks_PNG_formatted/opponent_right_label.png")


    blank_image.paste(opp_visual_right, (0, 0))
    blank_image.paste(opp_visual_left, (blank_image.size[0] - opp_visual_right.size[0], 0))

    blank_image.paste(my_left, (0, blank_image.size[1] - opp_visual_right.size[1]))
    blank_image.paste(my_right, (blank_image.size[0] - my_right.size[0], blank_image.size[1] - opp_visual_right.size[1]))

    blank_image.paste(my_left_label, (250, blank_image.size[1] - opp_visual_right.size[1] + 500))
    blank_image.paste(opponent_left_label, (250, opp_visual_right.size[1] - 550))

    blank_image.paste(my_right_label, (blank_image.size[0] - my_right.size[0] + 250, blank_image.size[1] - opp_visual_right.size[1] + 500))
    blank_image.paste(opponent_right_label, (blank_image.size[0] - my_right.size[0] + 250, opp_visual_right.size[1] - 550))
    # draw = ImageDraw.Draw(blank_image)
    # draw.line([(800,0),(800,1600)], fill=256, width=5)
    # del draw
    location = "../assets/PNG_generated_usertests/" + str(left) + "_" + str(right) + "_" + str(op_left) + "_" + str(op_right) +'.png'
    blank_image.save(location)

    return location


def generate_image_emergency(state):
    #[alert_and_able_to_move, breathing, open_airway, respiratory_rate, capillary_refill, pulse_rate]

    background = Image.new('RGBA', (1093, 719), 'white')
    chart = Image.open("../assets/Emergency_PNG_formatted/triage_chart.png")
    background.paste(chart, (0,0), mask=chart)

    d = ImageDraw.Draw(background)

    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 56)

    text_color = (100, 100, 200)
    if state[0]:
        d.text((990,100), "Yes", font=fnt, fill=text_color)
    else:
        d.text((990,100), "No", font=fnt, fill=text_color)

    if state[1]:
        d.text((990,210), "Yes", font=fnt, fill=text_color)
    else:
        d.text((990,210), "No", font=fnt, fill=text_color)

    d.text((990,320), "Yes", font=fnt, fill=text_color)

    d.text((990,430), str(state[3]), font=fnt, fill=text_color)
    d.text((990,540), str(state[4]), font=fnt, fill=text_color)
    d.text((990,650), str(state[5]), font=fnt, fill=text_color)

    location = ""
    for i in range(0, len(state)):
        location += str(state[i]) + "_"

    location = "../assets/PNG_generated_usertests/" + location + '.png'
    background.save(location)

    return location

def generate_image_highway(state):
    #State: [[left],[front, me, behind],[right],[exit#]]; 0 empty, 1 obstacle, 2 me, 41 - before my exit, 42 - my exit
    car_left = state[0][0]
    car_right = state[2][0]
    car_front = state[1][0]
    car_behind = state[1][2]
    car_self = state[1][1]
    exit_sign = state[3][0]

    basewidth = 2000
    carwidth = 300

    background = Image.open("../assets/Cars_PNG_formatted/background.png")
    wpercent = (basewidth/float(background.size[0]))
    hsize = int((float(background.size[1])*float(wpercent)))
    background = background.resize((basewidth,hsize), Image.ANTIALIAS)
    # print (background.size)

    # add your own car
    car_self_im = Image.open("../assets/Cars_PNG_formatted/Your_Car.png")
    wpercent = (carwidth/float(car_self_im.size[0]))
    hsize = int((float(car_self_im.size[1])*float(wpercent)))
    car_self_im = car_self_im.resize((carwidth,hsize), Image.ANTIALIAS)
    background.paste(car_self_im, (850,925), mask=car_self_im)

    other_car = Image.open("../assets/Cars_PNG_formatted/Other_Car.png")
    wpercent = (carwidth/float(other_car.size[0]))
    hsize = int((float(other_car.size[1])*float(wpercent)))
    other_car = other_car.resize((carwidth,hsize), Image.ANTIALIAS)

    if car_left == 1:
        background.paste(other_car, (475,800), mask=other_car)
    if car_front == 1:
        background.paste(other_car, (850,100), mask=other_car)
    if car_behind == 1:
        background.paste(other_car, (850,1600), mask=other_car)
    if car_right == 1:
        background.paste(other_car, (1225,1000), mask=other_car)
    if exit_sign == 42:
        exit_sign_im = Image.open("../assets/Cars_PNG_formatted/exit42.png")
        background.paste(exit_sign_im, (1625,650), mask=exit_sign_im)
    if exit_sign == 41:
        exit_sign_im = Image.open("../assets/Cars_PNG_formatted/exit41.png")
        background.paste(exit_sign_im, (1625,650), mask=exit_sign_im)


    if random.random() < 0.5:
        background.paste(other_car, (1225, -200), mask=other_car)
    location = "../assets/PNG_generated_usertests/car" + str(car_left) + "_" + str(car_front) + "_" + str(car_behind) + "_" + str(car_right) + "_" + str(exit_sign) +'.png'
    background.save(location)
    return location

def return_to_natural_language(str_to_convert, predicates):
    processed_string = ""
    remove_paren = 0
    i = 0

    while i < len(str_to_convert):
        if str_to_convert[i:i+len("not( ")] == "not( ":
            processed_string += "-"
            remove_paren += 1
            i += len("not( ")
        elif str_to_convert[i:i+len("not(")] == "not(":
            processed_string += "-"
            remove_paren += 1
            i += len("not(")
        elif remove_paren > 0 and str_to_convert[i] == ")":
            remove_paren -= 1
            i += 1
        else:
            processed_string += str_to_convert[i]
            i += 1

    predicate_map = generate_predicate_map(predicates)
    variables = re.findall('-?\d+', processed_string)
    for variable in variables:
        for predicate, predicate_value in predicate_map.items():

            predicate = predicate.replace("42", " forty-two ").replace("41", " forty-one ")
            predicate = predicate.replace("10", " ten ").replace("30", " thirty ")
            predicate = predicate.replace("40", " forty ").replace("70", " seventy ").replace("2"," two ")

            if abs(int(variable)) == predicate_value:
                if int(variable) < 0:
                    if " cannot " in predicate and " either " in predicate:
                        predicate = predicate.replace(" cannot ", " can ", 1)
                        predicate = predicate.replace(" either ", " one or both ", 1)
                    if " can " in predicate:
                        predicate = predicate.replace(" can ", " cannot ", 1)
                    if " is " in predicate:
                        predicate = predicate.replace(" is ", " is not ", 1)
                    if " outs " in predicate:
                        predicate = predicate.replace(" outs ", " does not out ", 1)
                    if " takes " in predicate:
                        predicate = predicate.replace(" takes ", " does not take ", 1)
                    if " have " in predicate:
                        predicate = predicate.replace(" have ", " do not have ", 1)

                processed_string = processed_string.replace(str(variable), predicate, 1)
                break
    processed_string = processed_string.replace("respiration count", "respiratory rate")
    processed_string = processed_string.replace(" forty-two ", "42")
    processed_string = processed_string.replace(" forty-one ", "41")
    processed_string = processed_string.replace(" two ", "2")
    processed_string = processed_string.replace(" ten ", "10")
    processed_string = processed_string.replace(" thirty ", "30")
    processed_string = processed_string.replace(" forty ", "40")
    processed_string = processed_string.replace(" seventy ", "70")
    return processed_string

def random_state_generator(domain):
    """
    Generate a state based on the given domain.
    Emergency (example: [0,1,1,13,1.8,68]):
        [
        0: alert + able to move,
        1: breathing,
        2: breathing after opening airway,
        3: respiration count,
        4: capillary refill time,
        5: pulse
        ]
    Highway (example: [[1][0,1][1][41]]:
        [
            [
            0,0: vehicle to my left
            ]
            [
            1,0: vehicle in front of me
            1,1: vehicle behind me
            ]
            [
            2,0: vehicle to my right
            ]
            [
            3,0: exit number
            ]
        ]
    Chopstics (example: [[0,4][1,3]]):
        [
            [
            0,0: my left hand
            0,1: my right hand
            ]
            [
            1,0: opponent's left
            1,1: opponent's right
            ]
        ]

    Parameters
    ----------
    domain : string
        Indicate which domain to generate a state for (e.g. "emergency")

    Return
    ------
    state : list
    """
    state = None
    if domain == "emergency":
        state = [0, random.randint(0,1), random.randint(0,1), random.randint(5, 45), round(random.uniform(0.5, 3.5),1) , random.randint(10,100)]
    elif domain == "highway":
        state = [[random.randint(0,1)],[random.randint(0,1),2,random.randint(0,1)],[random.randint(0,1)],[random.randint(41,42)]]
    elif domain == "chopsticks":
        my_left = random.randint(0,4)
        opp_left = random.randint(0,4)
        my_right = random.randint(0,4)
        opp_right = random.randint(0,4)

        if my_left == my_right == 0:
            if random.random() < 0.5:
                my_left = random.randint(2,4)
            else:
                my_right = random.randint(2,4)

        if opp_left == opp_right == 0:
            if random.random() < 0.5:
                opp_left = random.randint(2,4)
            else:
                opp_right = random.randint(2,4)

        state = [[my_left, my_right],[opp_left, opp_right]]
    return state

def generate_study(directory, agent, languages, study_id, domain, truth_assignment):
    """
    Create a user study. Write a CSV file.
    For each language in each domain, generate a random state and select an action
    with the desired truth assignment.

    Parameters:
    -----------
    directory : string
        Where to save the study
    agent : string
        Should the agent be good or bad?
    languages : list
        A list of all languages to convert to
    study_id : string
        An identifier for this study
    domain : string
        "all", "emergency", "chopsticks", or "highway"
    truth_assignment : bool
        Whether the agent should take the action or not

    Returns
    -------
    None
        Writes to .csv file
    """
    directory = directory + agent + "/"
    file_exists = os.path.isfile('../' + study_id + '_evaluate_logic_interpretability.csv')

    with open('../' + study_id + '_evaluate_logic_interpretability.csv', mode='a') as csv_file:
        fieldnames = ['qid',
                      'domain',
                      'agent',
                      'state',
                      'image_loc',
                      'target_language',
                      'eps-inv',
                      'action',
                      'action_truth_value',
                      'explanation',
                      'explanation_html',
                      'languages_satisfied',
                      'epsilon_flips',
                      'num_predicates',
                      'circuit_size',]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for language in languages:
            invert = False
            if isinstance(language, tuple):
                invert = language[1]
                language = language[0]

            action = None

            while (action == None):
                state = random_state_generator(domain)
                predicate_values = evaluate_predicates(state, directory, domain)
                actions = possible_actions(predicate_values, directory + "questions-formatted.txt")
                action = random_action(actions, truth_assignment)


            languages_satisfied = []

            epsilon_flips = 0

            statement_size = 0
            # generate explanation(s) for true action
            circuit = circuit_io.parse_NNF_DIMACS(directory + str(action) + ".nnf")
            circuit = formula_conversions.convert_to_given_language(circuit, language, invert)
            num_predicates = circuit.graph.get_variables(circuit.graph.find_dag_root())
            logic_formulas = action_natural_language_map[str(action)] + " when " + circuit.graph.collapse_graph_to_formula(circuit.graph.find_dag_root(), True).replace("  ", " ").replace("  ", " ").replace("( ", "(").replace(" )", ")")
            logic_formulas_html = action_natural_language_map[str(action)] + " when " + circuit.graph.collapse_graph_to_bullet_list().replace("  ", " ").replace("  ", " ").replace("( ", "(").replace(" )", ")")

            languages_satisfied.append( (action, circuit.languages) )
            epsilon_flips = len( [node for node in circuit.graph.nodes() if circuit.graph.nodes[node]['value'] in ['nand','nor'] ] )
            statement_size = circuit.num_nodes

            # true_circuit.graph.graph_logic_formula()
            # false_circuit.graph.graph_logic_formula()

            if domain == "chopsticks":
                image_location = generate_image_chopsticks(state)
            elif domain == "highway":
                image_location = generate_image_highway(state)
            elif domain == "emergency":
                image_location = generate_image_emergency(state)

            natural_flatsentence = return_to_natural_language(logic_formulas,
                                                                    directory + "predicates.txt")
            natural_htmlsentence = return_to_natural_language(logic_formulas_html,
                                                                    directory + "predicates.txt")

            writer.writerow({'qid': str(domain) + "_" + str(agent) + "_" + str(language) + "_" + str(invert) + "_truth" + str(truth_assignment),
                             'domain': str(domain),
                             'agent': str(agent),
                             'state': str(state),
                             'image_loc': str(image_location),
                             'target_language': str(language),
                             'eps-inv' : str(invert),
                             'action': str(action),
                             'action_truth_value': str(truth_assignment),
                             'explanation': str(natural_flatsentence),
                             'explanation_html': str(natural_htmlsentence),
                             'languages_satisfied': str(languages_satisfied),
                             'epsilon_flips': str(epsilon_flips),
                             'num_predicates': str(len(num_predicates)),
                             'circuit_size': str(statement_size)
                            })


def main():
    parser = argparse.ArgumentParser(description='Which domain.')
    parser.add_argument("-d", "--domain", default="chopsticks", required=False, help="Which domain?")
    args = vars(parser.parse_args())

    languages_to_convert_to = [(circuits.Languages.CNF, True),
                               (circuits.Languages.dDNNF, True),
                               (circuits.Languages.DNF, True),
                                circuits.Languages.dDNNF,
                                #circuits.Languages.sdDNNF,
                                circuits.Languages.CNF,
                                circuits.Languages.DNF,
                                circuits.Languages.OBDD,
                                # circuits.Languages.MODS,
                                circuits.Languages.ODNF]


    languages_to_convert_to_subset = [(circuits.Languages.CNF, True),
                                       circuits.Languages.dDNNF,
                                       circuits.Languages.DNF,
                                       circuits.Languages.ODNF]

    if args['domain'] == "chopsticks":
        directory = "../domains/chopsticks_examples/"
        # generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], True)
        generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], False)
        # generate_study(directory, "bad_agent", languages_to_convert_to_subset, "1", "chopsticks", False)

    elif args['domain'] == "highway":
        directory = "../domains/highway_examples/"
        generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], True)
        generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], False)
    elif args['domain'] == 'emergency':
        directory = "../domains/emergency_triage_examples/"
        generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], True)
        generate_study(directory, "good_agent", languages_to_convert_to, "1", args["domain"], False)
        generate_study(directory, "bad_agent", languages_to_convert_to_subset, "1", "emergency", True)

    elif args['domain'] == "all":
        directory = "../domains/chopsticks_examples/"
        study_id = random.getrandbits(32)
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "chopsticks", True)
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "chopsticks", False)
        generate_study(directory, "bad_agent", languages_to_convert_to_subset, str(study_id), "chopsticks", True)

        directory = "../domains/highway_examples/"
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "highway", True)
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "highway", False)
        generate_study(directory, "bad_agent", languages_to_convert_to_subset, str(study_id), "highway", True)

        directory = "../domains/emergency_triage_examples/"
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "emergency", True)
        generate_study(directory, "good_agent", languages_to_convert_to, str(study_id), "emergency", False)
        generate_study(directory, "bad_agent", languages_to_convert_to_subset, str(study_id), "emergency", True)

    elif args['domain'] == "test":
        state = [0, 1, 1, 14, 1.7, 39]


        predicate_values = evaluate_predicates(state, "../domains/emergency_triage_examples/good_agent/", "emergency")
        print (predicate_values)
        actions = possible_actions(predicate_values, "../domains/emergency_triage_examples/good_agent/questions-formatted.txt")
        print (actions)
        action = random_action(actions, False)
        print(action)
        circuit = circuit_io.parse_NNF_DIMACS("../domains/emergency_triage_examples/good_agent/immediate.nnf")
        circuit.graph.graph_logic_formula()
        circuit_a = formula_conversions.convert_to_given_language(circuit, circuits.Languages.CNF, False)
        circuit_a.graph.graph_logic_formula()
        circuit = formula_conversions.convert_to_given_language(circuit, circuits.Languages.CNF, True)
        circuit.graph.graph_logic_formula()
        print (circuit.graph.collapse_graph_to_formula(circuit.graph.find_dag_root()))
        print (circuit.graph.collapse_graph_to_formula(circuit.graph.find_dag_root(), True))

    elif args['domain'] == "test2":
        state = [0,1,1,17,2.1,68]
        generate_image_emergency(state)

if __name__ == "__main__":
    main()
