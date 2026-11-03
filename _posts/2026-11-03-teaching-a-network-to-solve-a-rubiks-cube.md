---
layout: post
title: Teaching a Neural Network to Solve a Rubik's Cube
subtitle: DeepCube vs. a beginner-method solver
tags: [machine learning, reinforcement learning, rubiks cube]
gh-repo: kjjoe/rubiks
gh-badge: [star, follow]
---

I trained a neural network to solve a Rubik's cube without ever showing it a solution, and compared it with the beginner-method solver I wrote in MATLAB a few years ago. Try both below.

{% include rubiks-cube.html %}

Turn faces with the buttons or the U, D, L, R, F, B keys (hold Shift for a counter-clockwise turn), scramble the cube, then pick a solver. DeepCube runs entirely in your browser and works best on scrambles of up to about 12 moves; the beginner method always finds a solution, just a long one.

The code is on [GitHub](https://github.com/kjjoe/rubiks).

<!--
Outline to write later:
- Background: the MATLAB beginner solver
- How DeepCube learns: value iteration from "solved = 0"
- Training: 5 minutes vs 1 hour, and the stall
- Results: solve rate and solution length vs the beginner method
- What's next: longer training, Kociemba's two-phase algorithm
-->
