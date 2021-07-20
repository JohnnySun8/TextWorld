# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


"""
.. _the_cooking_game:

The Cooking Game
================

This type of game was used for the competition *First TextWorld Problems* [1]_.
The overall objective of the game is to locate the kitchen, read the cookbook,
fetch the recipe's ingredients, process them accordingly, prepare the meal, and
eat it. To control the game's difficulty, one can specify the amount of skills
that are involved to solve it (see skills section below).

References
----------
.. [1] https://aka.ms/ftwp
"""

import os
import itertools
import textwrap
import argparse
from os.path import join as pjoin

from typing import Mapping, Dict, Optional, List, Tuple

import numpy as np
import networkx as nx
from numpy.random import RandomState

import textworld
from textworld.logic import Proposition
from textworld.generator.maker import WorldEntity
from textworld.generator.game import Quest, Event, GameOptions
from textworld.generator.data import KnowledgeBase

from textworld.utils import encode_seeds

from textworld.challenges import register

HELLO = ["Hello", "there!", "\n", "Did", "it", "work?"]