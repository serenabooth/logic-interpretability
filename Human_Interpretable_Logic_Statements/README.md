
# Add a domain
If you want to generate explanations, see the directory `Explanation_Generator` instead.

If you have already generated DNF explanation and want to apply knowledge compilation, first you will need to format the questions and responses and generate a predicate map by running `convert_questions.py`.

Formatted questions should take the form <action>: <DNF representation>. For example:
* redistribute: ((1 and -2 and -3 and -4 and -5))
* add_left_left: ((-3 and -4 and -5 ) or ( 2 and -4 ) or ( -1 and 2))

See `domains/highway_examples/good_agent/questions-formatted.txt`

For each action, we then create a file with the name action and the DIMACS representation of the logical explanation. See `FormulaFormats.pdf` for an explanation of these formats. And, yes, we know they could be a lot friendlier to use...

`compile_user_tests.py` generates a CSV of questions to present to users.

# Apply knowledge compilation and generate a user test!

Once you have added your domains, add code to `code/compile_user_tests.py` to (1) lookup your formulas, (2) generate possibly random representative states, and (3) create supporting graphics.

If you run `code/compile_user_tests.py -d all`, you generate a user test as described in the associated IJCAI 2019 publication. This will create a `.csv` file, as well as supporting graphics in `assets\PNG_generated_usertests\`.
