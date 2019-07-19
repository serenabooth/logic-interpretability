# Logic Interpretability

## More details and paper

http://slbooth.com/logic_interpretability.html

## Abstract

Knowledge compilation techniques translate propositional theories into equivalent forms to increase their computational tractability. But, how should we best present these propositional theories to a human? We analyze the standard taxonomy of propositional theories for relative *interpretability* across three model domains: highway driving, emergency triage, and the chopsticks game. We generate decision-making agents which produce logical explanations for their actions and apply knowledge compilation to these explanations. Then, we evaluate how quickly, accurately, and confidently users comprehend the generated explanations. We find that domain, formula size, and negated logical connectives significantly affect comprehension while formula properties typically associated with interpretability are not strong predictors of human ability to comprehend the theory.

## Code

For explanation generation, see `Explanation_Generator/`.
For knowledge compilation language conversions and user test generation, see `Human_Interpretable_Logic_Statements/`.

## Study Materials

See `StudyProcedures` for materials including: a PDF of all questions presented to study participants, a script for starting the user study, and a follow up discussion.

## Citation

@inproceedings{booth19:logic_interpretability,  
&emsp;  title = {Evaluating the Interpretability of the Knowledge Compilation Map:  
&emsp;  Communicating Logical Statements Effectively},  
&emsp;  author = {Serena Booth and Christian Muise and Julie Shah},  
&emsp;  booktitle = {IJCAI},  
&emsp;  year = {2019},  
}

## Authors
Serena Booth<sup>1</sup>, Christian Muise<sup>2,3</sup>, Julie Shah<sup>1</sup>

<sup>1</sup>MIT Computer Science and Artificial Intelligence Laboratory  
<sup>2</sup>MIT-IBM Watson AI Lab  
<sup>3</sup>IBM Research  

{serenabooth, julie_a_shah} [at] csail.mit.edu, christian.muise [at] ibm.com
