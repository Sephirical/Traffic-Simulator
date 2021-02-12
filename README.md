# COMP9417 Project - Traffic Light Controller

To use this program, compressions_style on line 462 can be either "default", "greedy" or "average" to generate different state representations 
for Q-learning to occur.
scheme on line 463 can either be "default" or "max_stopped" to change how the state is rewarded.
learning on line 464 can be either "default" or "fixed" to swap between allowing the program to decide when to switch lights as opposed to
switching the light at a fixed rate of once every 10 timesteps.
The Q-learning parameters gamma, alpha and epsilon can be changed in lines 21-23 and the generation function can be changed via
intensity on line 26 and generation_function on line 27 which can be either "default", "exponential" or "constant".
To be able to view a graphical version of the program, verbose needs to be set to True on line 42.
