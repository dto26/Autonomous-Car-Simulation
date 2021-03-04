# Autonomous Car Simulation

<img title="ai-car-simulation" alt="simulation" src="/resources/ai-car.png" width=400>

### Introduction

This is a simulation of how an autonomous vehicle can utilize reinforcement learning to navigate a road.
Pytorch is used for the Neural Network and Deep Q Learning. The rectangle represents the car while the three
dots represent sensors on the left, right, and front of the car. The yellow lines represent "sand".
Sand is the negative reinforcement. The top left corner and the bottom right corner are the positive reinforcements.
The goal of this vehicle is to search for the optimal path between these two locations. The car is designed to respond
to changes in the structure of the road and obstacles that may be in the way.

### Requirements

Make sure to have kivy installed. You can find the installation instructions [here](https://kivy.org/doc/stable/gettingstarted/installation.html#install-pip)
You will also need Python 3, pip, and PyTorch. (Python 3 is installed by default on Linux)

### Install Requirements (Linux)

```
$ sudo apt install python3-pip
$ pip install torch torchvision torchaudio
```

### Star The Simulation

```
python3 map.py
```

The car will start in an empty map. It will spin around in circles and move like an insect.
After a minute or two it will start moving between the top left and bottom right corner
more efficiently. I sugest saving the brain of the car at this point, it can be loaded again later.

<img title="ai-car-simulation" alt="simulation" src="/resources/car-empty-map.png" width=400>

The user can draw "sand" anywhere on the screen. It can be used to create an obstacle course
for the car to learn to navigate.

<img title="ai-car-simulation" alt="simulation" src="/resources/obstacle-course.png" width=400>
