# Explanations

This code is a partial implementation of *Hayes, Bradley, and Julie A. Shah. "Improving robot controller transparency through autonomous policy explanation." 2017 12th ACM/IEEE International Conference on Human-Robot Interaction, 2017.* Credit to Brad Hayes for most of this code.

To execute, write a simulator (such as `Simulation/simulator_highway_driving.py`) which records traces of its execution. Then, modify `Code/pcca_server.py` to specify where to look for your traces. Finally, execute `Code/pcca_server.py`. You can then query for explanations; for an example, see `get_explanations_highway.py`.
