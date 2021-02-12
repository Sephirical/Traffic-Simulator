from graphics import *
import numpy as np
import random
import time
import matplotlib.pyplot as plt

##############################################################################
# COMP9417 19T2 Project - Reinforcement Learning for Predictive Traffic Lights
# Authors:
# Jennifer Choi - z5115982
# Joseph Tran - z5115092
# Yiren Lin - z5118737
# Description:
# Using Q-Learning to operate the switching of a traffic light at an intersection
# and observing the impact of changing different simulation parameters on the
# implemented q-learning algorithm
##############################################################################

# Defined Parameters
# For q-learning
gamma = 0.9 # discount factor
alpha = 0.1 # learning rate
epsilon = 0.1  # chance to deviate and perform a random action

# Generation parameters
intensity = 10                   # determines frequency of car generation
generation_function = "default"  # exponential, constant

# Magic Numbers
GREEN = 0           # GREEN horizontal == RED vertical
YELLOW = 1          # YELLOW horizontal == RED_Next vertical
RED = 2             # RED,RED_NEXT have same functionality // similarly for GREEN, YELLOW
RED_NEXT = 3        # Suppose Vertical = GREEN -> YELLOW -> RED -> RED_NEXT -> GREEN ... 
                    # Then  Horizontal = RED -> RED_NEXT -> GREEN -> YELLOW -> RED ...
STAY = 0
SWITCH = 1          # switch means switch to next colour red->green->yellow->red

BLANK = 0           # Representation of block values
CAR = 1

# For simulator engine
verbose = False   # draw graphics to debug if true

if verbose == True:
    win = GraphWin("Traffic Light Simulator", 500, 500)

class Block:
    def __init__(self, x_co, y_co, size, colour):
        self.block = Rectangle(Point(x_co, y_co), Point(x_co + size, y_co + size))
        self.block.setFill(colour)
        self.block.draw(win)
    def changeColour(self, colour):
        self.block.undraw()
        self.block.setFill(colour)
        self.block.draw(win)
    def changeOutline(self, colour):
        self.block.setOutline(colour)

class Board:
    # Constructor
    def __init__(self,road_length, light_state, light_delay):
        self.h_left = []
        self.h_right = []
        self.v_up = []
        self.v_down = []
        self.h_left_graphics = []
        self.h_right_graphics = []
        self.v_up_graphics = []
        self.v_down_graphics = []
        self.light_state = light_state
        self.light_delay = light_delay
        self.timer = 0
        self.road_length = road_length
        side_size = 5
        if verbose == True:
            self.light_leftGraphic = Block(200, 230, 20, "green")
            self.light_upGraphic = Block(230, 200, 20, "red")
            self.light_rightGraphic = Block(290, 260, 20, "green")
            self.light_downGraphic = Block(260, 290, 20, "red")
        for i in range(self.road_length + 2):
            self.h_left.append(0)
            self.h_right.append(0)
            self.v_up.append(0)
            self.v_down.append(0)
            
            if verbose == True:
                temp = Block(i*side_size, (int(self.road_length/2))*side_size, side_size, "black")
                self.h_left_graphics.append(temp)
                
                temp = Block(i*side_size, (int(self.road_length/2) + 1)*side_size, side_size, "black")
                self.h_right_graphics.append(temp)
                
                temp = Block((int(self.road_length/2))*side_size, i*side_size, side_size, "black")
                self.v_up_graphics.append(temp)
                
                temp = Block((int(self.road_length/2) + 1)*side_size, i*side_size, side_size, "black")
                self.v_down_graphics.append(temp)
        
        
    # Methods
    # Return a board object after resetting to a defined initial state
    def reset(self):
        self.h_left.clear()
        self.h_right.clear()
        self.v_up.clear()
        self.v_down.clear()
        for i in range(self.road_length + 2):
            self.h_left.append(0)
            self.h_right.append(0)
            self.v_up.append(0)
            self.v_down.append(0)
            if verbose == True:
                self.light_leftGraphic.changeColour("green")
                self.light_rightGraphic.changeColour("green")
                self.light_upGraphic.changeColour("red")
                self.light_downGraphic.changeColour("red")
                self.h_left_graphics[i].changeColour("black")
                self.h_right_graphics[i].changeColour("black")
                self.v_up_graphics[i].changeColour("black")
                self.v_down_graphics[i].changeColour("black")
        self.light_state = GREEN
        self.light_delay = 0
        self.timer = 0
    
    # Determine whether or not to generate a car given a generation function and intensity value
    # Return value of incoming block
    def car_generating_function(self, intensity, function):
        next_tile = BLANK
        if function == "exponential":
            # generate a new car following an exponential distribution
            if random.expovariate(1/intensity) >= intensity:
                next_tile = CAR        
        elif function == "constant":
           # generate a new car at a constant rate
           # every [intensity] timesteps
            if self.timer % intensity == 0:
                next_tile = CAR        
        else: # default = random
            # generate a new car at a random rate
            # as intensity increases, the rate slows
            if self.timer % (random.randint(0, intensity) + 5) == 0:
                next_tile = CAR            
        return next_tile

    # Return the state size of the current board given a style of representation
    def get_state_size(self, style):
        if (style == "greedy"):
            # amount of cars in first [5] positions on horizontal roads
            # amount of cars in first [5] positions on vertical roads
            cars_h = 12
            cars_v = 12
            
        elif (style == "average"):
            # floor of sum of closest cars in distance [5] for horizontalRoads / 2
            # floor of sum of the closest cars in distance [5] for verticalRoads / 2
            cars_h = 6
            cars_v = 6
        else: # default = given
            # closest car from intersection for roads1 0-8 [9 if none]
            # closest car from intersection for roads2 0-8 [9 if none]
            cars_h = 10
            cars_v = 10
        
        # Common features
        # light setting 0-2 [0 = green, 1 = yellow, 2 = red 3 = red_next]
        # light delay 0-3
        light_setting = 4
        light_delay = 4
            
        return cars_h * cars_v * light_setting * light_delay
    
    # Return the state number of the board given the representation style
    def get_state_number(self, style):
        
        if (style == "greedy"):
            cars_road1 = sum(self.h_left[(int(self.road_length/2) - 6):(int(self.road_length/2) - 1)]) + \
            sum(self.h_right[(int(self.road_length/2) + 2):(int(self.road_length/2) + 7)])
            cars_road2 = sum(self.v_up[(int(self.road_length/2) - 6):(int(self.road_length/2) - 1)]) + \
            sum(self.v_down[(int(self.road_length/2) + 2):(int(self.road_length/2) + 7)])
            state_number = cars_road1 * 192 + cars_road2 * 16
    
        elif (style == "average"):
                
            cars_road1 = int((sum(self.h_left[(int(self.road_length/2) - 6):(int(self.road_length/2) - 1)]) + \
            sum(self.h_right[(int(self.road_length/2) + 2):(int(self.road_length/2) + 7)]))/2)
            cars_road2 = int((sum(self.v_up[(int(self.road_length/2) - 6):(int(self.road_length/2) - 1)]) + \
            sum(self.v_down[(int(self.road_length/2) + 2):(int(self.road_length/2) + 7)]))/2)
            state_number = cars_road1 * 96 + cars_road2 * 16
            
        else: # default
            h_left_score = 0
            for i in range(int(self.road_length/2) - 1, -1, -1):
                if self.h_left[i] == CAR:
                    break
                h_left_score += 1
            
            h_right_score = 0
            for i in range(int(self.road_length/2) + 2, self.road_length + 2):
                if self.h_right[i] == CAR:
                    break
                h_right_score += 1
            
            v_up_score = 0
            for i in range(int(self.road_length/2) - 1, -1, -1):
                if self.v_up[i] == CAR:
                    break
                v_up_score += 1
                
            v_down_score = 0
            for i in range(int(self.road_length/2) + 2, self.road_length + 2):
                if self.v_down[i] == CAR:
                    break
                v_down_score += 1
            
            pos1 = min(h_left_score, h_right_score, 9)
            pos2 = min(v_up_score, v_down_score, 9)
            state_number = pos1 * 160 + pos2 * 16
            #print(f"{state_number} = pos1({pos1}) * 160 + pos2({pos2}) * 16 + delay({self.light_delay}) * 4 + light({self.light_state})")
        
        # TODO: incorporate light states and delays    
        return state_number + self.light_delay * 4 + self.light_state
    
    # TODO: figure out array slices to do punishments!
    # Return the reward value of the current board state given a reward scheme
    def get_reward(self, scheme):
        reward = 0
        cars_stopped = 0
        if scheme == "default":
            # punish if there is any stopped car
            if (self.h_left[int(self.road_length/2) - 1] == CAR or self.h_right[int(self.road_length/2) + 2] == CAR):
                if (self.light_state == RED or self.light_state == YELLOW):
                    reward -= 1
            if (self.v_up[int(self.road_length/2) - 1] == CAR or self.v_down[int(self.road_length/2) + 2] == CAR):
                if (self.light_state == RED_NEXT or self.light_state == GREEN):
                    reward -= 1
            i = int(self.road_length/2) - 1
            while (self.h_left[i] == CAR and i > -1):
                i -= 1
                cars_stopped += 1
            i = int(self.road_length/2) + 2
            while (self.h_right[i] == CAR and i < self.road_length + 2):
                i += 1
                cars_stopped += 1
            i = int(self.road_length/2) - 1
            while (self.v_up[i] == CAR and i > -1):
                i -= 1
                cars_stopped += 1
            i = int(self.road_length/2) + 2
            while (self.v_down[i] == CAR and i < self.road_length + 2):
                i += 1
                cars_stopped += 1
        else: 
            # punish for number of stopped cars
            i = int(self.road_length/2) - 1
            while (self.h_left[i] == CAR and i > -1):
                i -= 1
                reward -= 1
            i = int(self.road_length/2) + 2
            while (self.h_right[i] == CAR and i < self.road_length + 2):
                i += 1
                reward -= 1
            i = int(self.road_length/2) - 1
            while (self.v_up[i] == CAR and i > -1):
                i -= 1
                reward -= 1
            i = int(self.road_length/2) + 2
            while (self.v_down[i] == CAR and i < self.road_length + 2):
                i += 1
                reward -= 1
            cars_stopped -= reward

        return cars_stopped, reward
    
    def act(self, action, reward_style, intensity, generation_function):
        self.timer += 1
        if self.light_delay > 0:
            self.light_delay -= 1
        else:
            self.light_delay = 0
        h_light = self.light_state
        # Perform associated action
        if action == SWITCH and self.light_delay == 0:
            #print(f"timer: {timer}, switched lights at intersection!")
            if self.light_state == YELLOW:
                if self.h_left[int(self.road_length/2)] != CAR and \
                self.h_left[int(self.road_length/2) - 1] != CAR and \
                self.h_right[int(self.road_length/2)] != CAR and \
                self.h_right[int(self.road_length/2) + 1] != CAR:
                    self.light_state = (self.light_state + 1) % 4
                    self.light_delay = 3
            elif self.light_state == RED_NEXT:
                if self.v_up[int(self.road_length/2)] != CAR and \
                self.v_up[int(self.road_length/2) - 1] != CAR and \
                self.v_down[int(self.road_length/2)] != CAR and \
                self.v_down[int(self.road_length/2) + 1] != CAR:
                    self.light_state = (self.light_state + 1) % 4
                    self.light_delay = 3
            else:
                self.light_state = (self.light_state + 1) % 4
                self.light_delay = 3
        # Advance timer by 1 tick
        if (h_light == RED):
            v_light = GREEN
        elif (h_light == RED_NEXT):
            v_light = YELLOW
        elif (h_light == GREEN):
            v_light = RED
        elif (h_light == YELLOW):
            v_light = RED_NEXT
    
        self.update_h_left(self.car_generating_function(intensity, generation_function), h_light)
        self.update_h_right(self.car_generating_function(intensity, generation_function), h_light)
        self.update_v_up(self.car_generating_function(intensity, generation_function), v_light)
        self.update_v_down(self.car_generating_function(intensity, generation_function), v_light)
        
        self.reward = self.get_reward(reward_style)
        
        # Slow down the animation for human checking        
        if verbose == True:
            time.sleep(0.5)
            self.update_graphics()
    
    def update_h_left(self, flag, light_state):
        if light_state == RED or light_state == RED_NEXT:
            # Check for the first empty space before intersection
            # Remove it then shift array
            for i in range(int(self.road_length/2) - 1, -1, -1):
                if self.h_left[i] == BLANK:
                    self.h_left.pop(i)
                    if flag == CAR:
                        self.h_left.insert(0, 1)
                    else:
                        self.h_left.insert(0, 0)
                    break
            self.h_left.pop()
            self.h_left.insert(int(self.road_length/2) + 2, 0)
        else:
            self.h_left.pop()
            if flag == CAR:
                self.h_left.insert(0, 1)
            else:
                self.h_left.insert(0, 0)
            self.v_up[int(self.road_length/2)] = self.h_left[int(self.road_length/2)]
            self.v_down[int(self.road_length/2)] = self.h_left[int(self.road_length/2) + 1]
            
    def update_h_right(self, flag, light_state):
        if light_state == RED or light_state == RED_NEXT:
            # Check for the first empty space before intersection
            # Remove it then shift array
            for i in range(int(self.road_length/2) + 2, self.road_length):
                if self.h_right[i] == BLANK:
                    self.h_right.pop(i)
                    if flag == CAR:
                        self.h_right.append(1)
                    else:
                        self.h_right.append(0)
                    break
            self.h_right.pop(0)
            self.h_right.insert(int(self.road_length/2) - 1, 0)
        else:
            self.h_right.pop(0)
            if flag == CAR:
                self.h_right.insert(self.road_length - 1, 1)
            else:
                self.h_right.insert(self.road_length - 1, 0)
            self.v_up[int(self.road_length/2) + 1] = self.h_right[int(self.road_length/2)]
            self.v_down[int(self.road_length/2) + 1] = self.h_right[int(self.road_length/2) + 1]
            
    def update_v_up(self, flag, light_state):
        if light_state == RED or light_state == RED_NEXT:
            # Check for the first empty space before intersection
            # Remove it then shift array
            for i in range(int(self.road_length/2) - 1, -1, -1):
                if self.v_up[i] == BLANK:
                    self.v_up.pop(i)
                    if flag == CAR:
                        self.v_up.insert(0, 1)
                    else:
                        self.v_up.insert(0, 0)
                    break
            self.v_up.pop()
            self.v_up.insert(int(self.road_length/2) + 2, 0)
        else:
            self.v_up.pop()
            if flag == CAR:
                self.v_up.insert(0, 1)
            else:
                self.v_up.insert(0, 0)
            self.h_left[int(self.road_length/2)] = self.v_up[int(self.road_length/2)]
            self.h_right[int(self.road_length/2)] = self.v_up[int(self.road_length/2) + 1]
            
    def update_v_down(self, flag, light_state):
        if light_state == RED or light_state == RED_NEXT:
            # Check for the first empty space before intersection
            # Remove it then shift array
            for i in range(int(self.road_length/2) + 2, self.road_length):
                if self.v_down[i] == BLANK:
                    self.v_down.pop(i)
                    if flag == CAR:
                        self.v_down.append(1)
                    else:
                        self.v_down.append(0)
                    break
            self.v_down.pop(0)
            self.v_down.insert(int(self.road_length/2) - 1, 0)
        else:
            self.v_down.pop(0)
            if flag == CAR:
                self.v_down.append(1)
            else:
                self.v_down.append(0)
            self.h_left[int(self.road_length/2) + 1] = self.v_down[int(self.road_length/2)]
            self.h_right[int(self.road_length/2) + 1] = self.v_down[int(self.road_length/2) + 1]
            
    def update_graphics(self):
        for i in range(self.road_length + 2):
            if self.h_left[i] == BLANK:
                self.h_left_graphics[i].changeColour("black")
            else:
                self.h_left_graphics[i].changeColour("white")
            
            if self.h_right[i] == BLANK:
                self.h_right_graphics[i].changeColour("black")
            else:
                self.h_right_graphics[i].changeColour("white")
            
            if self.v_up[i] == BLANK:
                self.v_up_graphics[i].changeColour("black")
            else:
                self.v_up_graphics[i].changeColour("white")
                
            if self.v_down[i] == BLANK:
                self.v_down_graphics[i].changeColour("black")
            else:
                self.v_down_graphics[i].changeColour("white")
        if self.light_state == RED:
            self.light_leftGraphic.changeColour("red")
            self.light_rightGraphic.changeColour("red")
            self.light_upGraphic.changeColour("green")
            self.light_downGraphic.changeColour("green")
        elif self.light_state == YELLOW:
            self.light_leftGraphic.changeColour("yellow")
            self.light_rightGraphic.changeColour("yellow")
            self.light_upGraphic.changeColour("red")
            self.light_downGraphic.changeColour("red")
        elif self.light_state == GREEN:
            self.light_leftGraphic.changeColour("green")
            self.light_rightGraphic.changeColour("green")
            self.light_upGraphic.changeColour("red")
            self.light_downGraphic.changeColour("red")
        else:
            self.light_leftGraphic.changeColour("red")
            self.light_rightGraphic.changeColour("red")
            self.light_upGraphic.changeColour("yellow")
            self.light_downGraphic.changeColour("yellow")

# Driver code
def main():
    board = Board(100, GREEN, 0)
    rewards = []
    stopped = []
    compression_style = "average"  # greedy/average/default
    scheme = "default"             # default/max_stopped
    learning = "default"
    state_size = board.get_state_size(compression_style)
    action_size = 2
    # Initialise Q table
    Q = np.zeros((state_size, action_size))
    # Evaluate agent performance after training
    # Run for 1000 episodes each of 10000 timesteps
    episodes = 1000
    timesteps = 10000
    for i in range(episodes):
        board.reset()
        total_reward = 0
        total_stopped = 0
        state = board.get_state_number(compression_style)
        for j in range(timesteps):
            if learning == "fixed":
                if j % 10 == 0:
                    action = SWITCH
                else:
                    action = STAY
            else:
                if (random.uniform(0,1) < epsilon):
                    # return a random action (0 or 1)
                    action = random.randrange(0,2)
                else:
                    # return best action according to table
                    action = np.argmax(Q[state])
            board.act(action, scheme, intensity, generation_function)
            cars_stopped, reward = board.get_reward(scheme)
            total_stopped += cars_stopped
            total_reward += reward
            next_state = board.get_state_number(compression_style)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
        #print(total_reward)
    # test performance after 1000 episodes
    for i in range(5):
        board.reset()
        total_stopped = 0
        state = board.get_state_number(compression_style)
        for j in range(timesteps):
            action = np.argmax(Q[state])
            board.act(action, scheme, intensity, generation_function)
            cars_stopped, reward = board.get_reward(scheme)
            total_stopped += cars_stopped
            next_state = board.get_state_number(compression_style)
            state = next_state
        stopped.append(float(total_stopped)/timesteps)
    print(stopped)
    if verbose == True:
        win.close()
            
if __name__ == '__main__':
    main()
    
