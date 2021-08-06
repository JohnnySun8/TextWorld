# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
.. _the_shopping_game:

The Shopping game
================

This type of game is similar to the cooking game used for the competition *First TextWorld Problems* [1]_.
The overall objective of the game is to obtain the items supplied in a shopping/grocery list. The items will be in different sections of the game world. The world changes with the number of rooms specified, adding more complexity. Additionally, to control the game's difficulty, one can specify the amount of skills
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


KB_PATH = pjoin(os.path.dirname(__file__), "textworld_data")
KB_LOGIC_PATH = pjoin(KB_PATH, "logic")
KB_LOGIC_DROP_PATH = pjoin(KB_PATH, "logic_drop")
KB_GRAMMAR_PATH = pjoin(KB_PATH, "text_grammars")

SKILLS = ["recipe", "take", "cook", "cut", "open", "drop", "go"]

FRESH_ADJECTIVES = ["fresh"]
ROTTEN_ADJECTIVES = ["rotten", "expired", "rancid"]

TYPES_OF_COOKING = ["raw", "fried", "roasted", "grilled"]
TYPES_OF_CUTTING = ["uncut", "chopped", "sliced", "diced"]

TYPES_OF_COOKING_VERBS = {"fried": "fry", "roasted": "roast", "grilled": "grill"}
TYPES_OF_CUTTING_VERBS = {"chopped": "chop", "sliced": "slice", "diced": "dice"}

FOODS_SPLITS = {
    'train': [
        # CLOTHING
        'jeans',
        't-shirt',
        'socks',
        'suit',
        'sweater',
        # HYGIENE
        'toothpaste',
        'toothbrush',
        'deodorant',
        'detergent',
        'sponge',
        # VEG
        'broccoli',
        'aubergine',
        'red radish',
        # FRUIT
        'mango',
        'plum',
        'white grapes',
        # MEAT
        'salmon',
        'pork bacon', 
        # STATIONERY
        'pen',
        'paper',
        'eraser',
        'scissors',
        'notebook',
        # TECH & HW
        'smartphone',
        'vacuum cleaner',
        'monitor',
        'kettle',
        'microwave',
        # INGR
        'condensed milk',
        'cumin powder',
        'dark chocolate',
        # REST
        'chicken alfredo',
        'meat shawarma',
        'tandoori chicken',
        'shrimp pizza',
        'corned beef',
        # ORIG
        'orange bell pepper',
        'block of cheese',
        'black pepper',
        'red hot pepper',
        'yellow bell pepper',
        'banana',
        'salt',
        'chicken leg',
        'cilantro',
        'white onion',
        'purple potato',
        'olive oil',
        'flour',
        'red onion',
        'yellow potato',
        'parsley',
        'red potato',
        'water',
        'pork chop',
        'red apple',
        'chicken wing',
        'carrot'        
    ],
    'valid': [
        # CLOTHING
        'shorts',
        'shirt',
        'jacket',
        'skirt',
        'bra',
        # HYGIENE
        'cotton balls',
        'q tips',
        'bar soap',
        'shampoo',
        'lotion',
        # VEG
        'spinach',
        'pink radish',
        # FRUIT
        'pineapple',
        'red grapes',
        # MEAT
        'beef bacon',
        'cod',
        # STATIONERY
        'paper clips',
        'stapler',
        'folder',
        # TECH & HW
        'coffee maker',
        'keyboard',
        'dehumidifier',
        # INGR
        'milk chocolate',
        'maple syrup',
        # REST
        'beef lasagne',
        'cheeseburger',
        'cottage pie',
        # ORIG
        'vegetable oil',
        'green apple',
        'red tuna',
        'green bell pepper',
        'red bell pepper',
        'lettuce',
        'peanut oil',
        'chicken breast'
    ],
    'test': [
        # CLOTHING
        'coat',
        'boots',
        'hoodie',
        'boxer shorts'
        'scarf',
        # HYGIENE
        'dental floss',
        'tissues',
        'broomstick',
        'conditioner',
        'hand sanitiser',
        # VEG
        'black radish',
        # FRUIT
        'black grapes',
        # MEAT
        'turkey bacon',
        # STATIONERY
        'pencil',
        'fountain pen',
        'highlighter',
        # TECH & HW
        'blender',
        'smartwatch'
        # INGR
        'white chocolate',
        'honey',
        # REST
        'shepherds pie'
        'fried chicken',
        # ORIG
        'milk',
        'yellow onion',
        'yellow apple',
        'sugar',
        'egg',
        'green hot pepper',
        'white tuna',
        'tomato'
    ],
}

FOOD_PREPARATIONS_SPLITS = {
    'train': {
         # CLOTHING
        'jeans': [
            ('raw', 'uncut')
        ],
        't-shirt': [
            ('raw', 'uncut')
        ],
        'socks': [
            ('raw', 'uncut')
        ],
        'suit': [
            ('raw', 'uncut')
        ],
        'sweater': [
            ('raw', 'uncut')
        ],
        'toothpaste': [
            ('raw', 'uncut')
        ],
        'toothbrush': [
            ('raw', 'uncut')
        ],
        'deodorant': [
            ('raw', 'uncut')
        ],
        'detergent': [
            ('raw', 'uncut')
        ],
        'sponge': [
            ('raw', 'uncut')
        ],
        'broccoli': [
            ('raw', 'uncut')
        ],
        'aubergine': [
            ('raw', 'uncut')
        ],
        'red radish': [
            ('raw', 'uncut')
        ],
        'mango': [
            ('raw', 'uncut')
        ],
        'plum': [
            ('raw', 'uncut')
        ],
        'white grapes': [
            ('raw', 'uncut')
        ],
        'salmon': [
            ('raw', 'uncut')
        ],
        'pork bacon': [
            ('raw', 'uncut')
        ],
        'pen': [
            ('raw', 'uncut')
        ],
        'paper': [
            ('raw', 'uncut')
        ],
        'eraser': [
            ('raw', 'uncut')
        ],
        'scissors': [
            ('raw', 'uncut')
        ],
        'notebook': [
            ('raw', 'uncut')
        ],
        'smartphone': [
            ('raw', 'uncut')
        ],
        'vacuum cleaner': [
            ('raw', 'uncut')
        ],
        'monitor': [
            ('raw', 'uncut')
        ],
        'kettle': [
            ('raw', 'uncut')
        ],
        'microwave': [
            ('raw', 'uncut')
        ],
        'condensed milk': [
            ('raw', 'uncut')
        ],
        'cumin powder': [
            ('raw', 'uncut')
        ],
        'dark chocolate': [
            ('raw', 'uncut')
        ],
        'chicken alfredo': [
            ('raw', 'uncut')
        ],
        'meat shawarma': [
            ('raw', 'uncut')
        ],
        'tandoori chicken': [
            ('raw', 'uncut')
        ],
        'shrimp pizza': [
            ('raw', 'uncut')
        ],
        'corned beef': [
            ('raw', 'uncut')
        ],
                
        # ORIGINAL COOOKING 
        'orange bell pepper': [
            ('raw', 'chopped'), ('roasted', 'diced'), ('grilled', 'uncut'), ('raw', 'uncut'), ('raw', 'sliced'),
            ('grilled', 'sliced'), ('roasted', 'sliced'), ('fried', 'diced'), ('grilled', 'chopped')
        ],
        'block of cheese': [
            ('fried', 'diced'), ('fried', 'uncut'), ('grilled', 'chopped'), ('raw', 'chopped'), ('grilled', 'diced'),
            ('roasted', 'chopped'), ('grilled', 'sliced'), ('raw', 'uncut'), ('raw', 'sliced')
        ],
        'black pepper': [
            ('raw', 'uncut')
        ],
        'red hot pepper': [
            ('roasted', 'sliced'), ('fried', 'chopped'), ('roasted', 'uncut'), ('fried', 'sliced'), ('raw', 'sliced'),
            ('grilled', 'chopped'), ('fried', 'uncut'), ('raw', 'chopped'), ('grilled', 'sliced')
        ],
        'yellow bell pepper': [
            ('roasted', 'chopped'), ('grilled', 'sliced'), ('fried', 'sliced'), ('raw', 'diced'), ('roasted', 'diced'),
            ('fried', 'chopped'), ('roasted', 'uncut'), ('grilled', 'uncut'), ('fried', 'uncut')
        ],
        'banana': [
            ('grilled', 'diced'), ('fried', 'chopped'), ('grilled', 'chopped'), ('grilled', 'sliced'), ('fried', 'diced'),
            ('roasted', 'diced'), ('fried', 'sliced'), ('raw', 'sliced'), ('roasted', 'sliced')
        ],
        'salt': [
            ('raw', 'uncut')
        ],
        'chicken leg': [
            ('grilled', 'uncut')
        ],
        'cilantro': [
            ('raw', 'uncut'), ('raw', 'diced')
        ],
        'white onion': [
            ('grilled', 'uncut'), ('raw', 'chopped'), ('roasted', 'uncut'), ('roasted', 'sliced'), ('fried', 'diced'),
            ('raw', 'sliced'), ('grilled', 'chopped'), ('roasted', 'chopped'), ('roasted', 'diced')
        ],
        'purple potato': [
            ('roasted', 'sliced'), ('roasted', 'diced'), ('grilled', 'diced'), ('fried', 'chopped'), ('fried', 'sliced'),
            ('fried', 'diced'), ('roasted', 'uncut')
        ],
        'olive oil': [
            ('raw', 'uncut')
        ],
        'flour': [
            ('raw', 'uncut')
        ],
        'red onion': [
            ('raw', 'uncut'), ('roasted', 'uncut'), ('roasted', 'diced'), ('fried', 'sliced'), ('raw', 'sliced'),
            ('grilled', 'diced'), ('fried', 'diced'), ('raw', 'diced'), ('grilled', 'sliced')
        ],
        'yellow potato': [
            ('grilled', 'chopped'), ('grilled', 'sliced'), ('fried', 'diced'), ('fried', 'sliced'), ('fried', 'chopped'),
            ('roasted', 'chopped'), ('roasted', 'uncut')
        ],
        'parsley': [
            ('raw', 'diced'), ('raw', 'sliced')
        ],
        'red potato': [
            ('roasted', 'sliced'), ('grilled', 'chopped'), ('fried', 'uncut'), ('fried', 'chopped'), ('fried', 'diced'),
            ('fried', 'sliced'), ('roasted', 'diced')
        ],
        'water': [
            ('raw', 'uncut')
        ],
        'pork chop': [
            ('fried', 'sliced'), ('roasted', 'sliced'), ('grilled', 'uncut'), ('roasted', 'diced'), ('grilled', 'diced'),
            ('fried', 'uncut'), ('fried', 'chopped')
        ],
        'red apple': [
            ('grilled', 'sliced'), ('fried', 'diced'), ('roasted', 'sliced'), ('fried', 'sliced'), ('grilled', 'diced'),
            ('raw', 'uncut'), ('raw', 'sliced'), ('raw', 'diced'), ('roasted', 'chopped')
        ],
        'chicken wing': [
            ('grilled', 'uncut')
        ],
        'carrot': [
            ('roasted', 'sliced'), ('fried', 'chopped'), ('raw', 'uncut'), ('grilled', 'uncut'), ('roasted', 'uncut'),
            ('grilled', 'sliced'), ('raw', 'sliced'), ('fried', 'sliced'), ('raw', 'chopped')
        ]},
    'valid': {
        'orange bell pepper': [('roasted', 'chopped'), ('fried', 'uncut'), ('fried', 'sliced'), ('raw', 'diced')],
        'block of cheese': [('roasted', 'diced'), ('grilled', 'uncut'), ('raw', 'diced'), ('roasted', 'sliced')],
        'black pepper': [('raw', 'uncut')],
        'red hot pepper': [('raw', 'diced'), ('roasted', 'chopped'), ('roasted', 'diced'), ('grilled', 'diced')],
        'yellow bell pepper': [('raw', 'chopped'), ('roasted', 'sliced'), ('fried', 'diced'), ('raw', 'sliced')],
        'banana': [('roasted', 'uncut'), ('grilled', 'uncut'), ('raw', 'diced'), ('roasted', 'chopped')],
        'salt': [('raw', 'uncut')],
        'chicken leg': [('fried', 'uncut')],
        'cilantro': [('raw', 'sliced')],
        'white onion': [('grilled', 'sliced'), ('raw', 'diced'), ('fried', 'chopped'), ('fried', 'uncut')],
        'purple potato': [('grilled', 'chopped'), ('grilled', 'uncut'), ('fried', 'uncut')],
        'olive oil': [('raw', 'uncut')],
        'flour': [('raw', 'uncut')],
        'red onion': [('roasted', 'chopped'), ('fried', 'chopped'), ('fried', 'uncut'), ('grilled', 'chopped')],
        'yellow potato': [('roasted', 'diced'), ('grilled', 'uncut'), ('grilled', 'diced')],
        'parsley': [('raw', 'uncut')],
        'red potato': [('grilled', 'diced'), ('grilled', 'sliced'), ('roasted', 'chopped')],
        'water': [('raw', 'uncut')],
        'pork chop': [('fried', 'diced'), ('roasted', 'chopped'), ('roasted', 'uncut')],
        'red apple': [('raw', 'chopped'), ('roasted', 'diced'), ('grilled', 'uncut'), ('fried', 'chopped')],
        'chicken wing': [('roasted', 'uncut')],
        'carrot': [('grilled', 'chopped'), ('fried', 'uncut'), ('roasted', 'chopped'), ('roasted', 'diced')]},
    'test': {
        'orange bell pepper': [('roasted', 'uncut'), ('fried', 'chopped'), ('grilled', 'diced')],
        'block of cheese': [('fried', 'chopped'), ('roasted', 'uncut'), ('fried', 'sliced')],
        'black pepper': [('raw', 'uncut')],
        'red hot pepper': [('raw', 'uncut'), ('grilled', 'uncut'), ('fried', 'diced')],
        'yellow bell pepper': [('grilled', 'chopped'), ('raw', 'uncut'), ('grilled', 'diced')],
        'banana': [('raw', 'chopped'), ('fried', 'uncut'), ('raw', 'uncut')],
        'salt': [('raw', 'uncut')],
        'chicken leg': [('roasted', 'uncut')],
        'cilantro': [('raw', 'chopped')],
        'white onion': [('raw', 'uncut'), ('fried', 'sliced'), ('grilled', 'diced')],
        'purple potato': [('grilled', 'sliced'), ('roasted', 'chopped')],
        'olive oil': [('raw', 'uncut')],
        'flour': [('raw', 'uncut')],
        'red onion': [('raw', 'chopped'), ('grilled', 'uncut'), ('roasted', 'sliced')],
        'yellow potato': [('fried', 'uncut'), ('roasted', 'sliced')],
        'parsley': [('raw', 'chopped')],
        'red potato': [('grilled', 'uncut'), ('roasted', 'uncut')],
        'water': [('raw', 'uncut')],
        'pork chop': [('grilled', 'sliced'), ('grilled', 'chopped')],
        'red apple': [('fried', 'uncut'), ('roasted', 'uncut'), ('grilled', 'chopped')],
        'chicken wing': [('fried', 'uncut')],
        'carrot': [('raw', 'diced'), ('grilled', 'diced'), ('fried', 'diced')]
    }
}

FOODS_COMPACT = {
    "jeans": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "t-shirt": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "socks": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "suit": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "sweater": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "shorts": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "shirt": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "jacket": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "skirt": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "bra": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "coat": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "boots": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "hoodie": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "boxer shorts": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "scarf": {
        "properties": ["inedible"],
        "locations": ["clothing.clothing showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "toothpaste": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "toothbrush": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "deodorant": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "detergent": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "sponge": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "cotton balls": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "q tips": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "bar soap": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "shampoo": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "lotion": {
        "properties": ["inedible"],
        "locations": ["hygiene.ingredients showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "dental floss": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "tissues": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "broomstick": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "conditioner": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "hand santiser": {
        "properties": ["inedible"],
        "locations": ["hygiene.hygiene showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "broccoli": {
        "properties": ["edible"],
        "locations": ["vegetables.green veg basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "aubergine": {
        "properties": ["edible"],
        "locations": ["vegetables.veg basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "red radish": {
        "properties": ["edible"],
        "locations": ["vegetables.veg fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "spinach": {
        "properties": ["edible"],
        "locations": ["vegetables.veg fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "pink radish": {
        "properties": ["edible"],
        "locations": ["vegetables.veg fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "black radish": {
        "properties": ["edible"],
        "locations": ["vegetables.veg fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "pink radish": {
        "properties": ["edible"],
        "locations": ["vegetables.veg fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "black grapes": {
        "properties": ["edible"],
        "locations": ["fruits.fruit basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "red grapes": {
        "properties": ["edible"],
        "locations": ["fruits.fruit basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "white grapes": {
        "properties": ["edible"],
        "locations": ["fruits.fruit basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "pineapple": {
        "properties": ["edible"],
        "locations": ["fruits.yellow basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "mango": {
        "properties": ["edible"],
        "locations": ["fruits.yellow basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "plum": {
        "properties": ["edible"],
        "locations": ["fruits.fruit basket", "checkout.basket"], #"supermarket.showcase"],
    },
    "turkey bacon": {
        "properties": ["edible"],
        "locations": ["meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "beef bacon": {
        "properties": ["edible"],
        "locations": ["meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "pork bacon": {
        "properties": ["edible"],
        "locations": ["meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "cod": {
        "properties": ["edible"],
        "locations": ["meats.meats showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "salmon": {
        "properties": ["edible"],
        "locations": ["meats.meats showcase", "checkout.showcase"], #"supermarket.showcase"],
    },
    "pen": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "paper": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "eraser": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "scissors": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "notebook": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "paper clips": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "stapler": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "folder": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "pencil": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "fountain pen": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "highlighter": {
        "properties": ["edible"],
        "locations": ["stationery.stationery shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "smartphone": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "vacuum cleaner": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "monitor": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "kettle": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "microwave": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "coffee maker": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "keyboard": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "dehumidifier": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "blender": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "smartwatch": {
        "properties": ["edible"],
        "locations": ["hardware.hardware shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "condensed milk": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "cumin powder": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "dark chocolate": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "milk chocolate": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "maple syrup": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "white chocolate": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "honey": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf", "checkout.shelf"], #"supermarket.showcase"],
    },
    "chicken alfredo": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "meat shawarma": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "tandoori chicken": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "shrimp pizza": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "corned beef": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "beef lasagne": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "cheeseburger": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "cottage pie": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "shepherds pie": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    "fried chicken": {
        "properties": ["edible"],
        "locations": ["restaurant.restaurant table", "checkout.shelf"], #"supermarket.showcase"],
    },
    
    # ORIGINAL COOKING 
    "egg": {
        "properties": ["inedible", "cookable", "needs_cooking"],
        "locations": ["ingredients.ingredients fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "milk": {
        "indefinite": "some",
        "properties": ["drinkable", "inedible"],
        "locations": ["ingredients.ingredients fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "water": {
        "indefinite": "some",
        "properties": ["drinkable", "inedible"],
        "locations": ["ingredients.ingredients fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "cooking oil": {
        "names": ["vegetable oil", "peanut oil", "olive oil"],
        "indefinite": "some",
        "properties": ["inedible"],
        "locations": ["ingredients.ingredients shelf"], #"supermarket.showcase"],
    },
    "chicken wing": {
        "properties": ["inedible", "cookable", "needs_cooking"],
        "locations": ["meats.meats showcase", "meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "chicken leg": {
        "properties": ["inedible", "cookable", "needs_cooking"],
        "locations": ["meats.meats showcase", "meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "chicken breast": {
        "properties": ["inedible", "cookable", "needs_cooking"],
        "locations": ["meats.meats showcase", "meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "pork chop": {
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut"],
        "locations": ["meats.meats showcase", "meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "tuna": {
        "names": ["red tuna", "white tuna"],
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut"],
        "locations": ["meats.meats showcase", "meats.meats fridge", "checkout.fridge"], #"supermarket.showcase"],
    },
    "carrot": {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.orange basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "onion": {
        "names": ["red onion", "white onion", "yellow onion"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "lettuce": {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.green veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "potato": {
        "names": ["red potato", "yellow potato", "purple potato"],
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut"],
        "locations": ["vegetables.veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "apple": {
        "names": ["red apple", "yellow apple", "green apple"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["fruits.fruit basket","fruits.fruits fridge", "checkout.fridge"], #"garden"],
    },
    "banana": {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["fruits.yellow basket","fruits.fruits fridge", "checkout.fridge"], #"garden"],
    },
    "tomato": {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.red veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "hot pepper": {
        "names": ["red hot pepper", "green hot pepper"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "bell pepper": {
        "names": ["red bell pepper", "yellow bell pepper", "green bell pepper", "orange bell pepper"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["vegetables.veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "black pepper": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf"], #pantry.shelf", "supermarket.showcase"],
    },
    "flour": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf"], #"pantry.shelf", "supermarket.showcase"],
    },
    "salt": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf"], #"pantry.shelf", "supermarket.showcase"],
    },
    "sugar": {
        "properties": ["edible"],
        "locations": ["ingredients.ingredients shelf"], #"pantry.shelf", "supermarket.showcase"],
    },
    "block of cheese": {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut"],
        "locations": ["ingredients.ingredients fridge", "ingredients.ingredients showcase", "checkout.fridge"], #"supermarket.showcase"],
    },
    "cilantro": {
        "properties": ["edible", "cuttable", "uncut"],
        "locations": ["vegetables.green veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    },
    "parsley": {
        "properties": ["edible", "cuttable", "uncut"],
        "locations": ["vegetables.green veg basket","vegetables.veg fridge", "checkout.fridge"], #"garden"],
    }
}

FOODS = {}
for k, v in FOODS_COMPACT.items():
    if "names" in v:
        for name in v["names"]:
            FOODS[name] = dict(v)
            del FOODS[name]["names"]
    else:
        FOODS[k] = v


ENTITIES = {
    "cookbook": {
        "type": "o",
        "names": ["cookbook", "recipe book"],
        "adjs": ["interesting"],
        "locations": ["checkout.counter"],
        "properties": [],
        "desc": [None],
    },
    "knife": {
        "type": "o",
        "names": ["knife"],
        "adjs": ["sharp"],
        "locations": ["checkout.counter", "hardware.hardware shelf"],
        "properties": ["sharp"],
        "desc": [None],
    },
    
     "shopping list": {
        "type": "o",
        "names": ["shopping list", "grocery list", "list"],
        "adjs": ["interesting"],
        "locations": ["checkout.counter"],
        "properties": [],
        "desc": [None],
    },

    # Kitchen
    "fridge": {
        "type": "c",
        "names": ["fridge", "refrigerator"],
        "adjs": ["conventional"],
        "locations": ["checkout"],
        "properties": ["closed"],
        "desc": [None],
    },
    "ingredients fridge": {
        "type": "c",
        "names": ["ingredients fridge", "ingredients refrigerator"],
        "adjs": ["conventional"],
        "locations": ["ingredients"],
        "properties": ["closed"],
        "desc": [None],
    },
    "veg fridge": {
        "type": "c",
        "names": ["veg fridge", "veg refrigerator"],
        "adjs": ["conventional"],
        "locations": ["vegetables",],
        "properties": ["closed"],
        "desc": [None],
    },
    "fruits fridge": {
        "type": "c",
        "names": ["fruits fridge", "fruits refrigerator"],
        "adjs": ["conventional"],
        "locations": ["fruits"],
        "properties": ["closed"],
        "desc": [None],
    },
    "meats fridge": {
        "type": "c",
        "names": ["meats fridge", "meats refrigerator", "meats freezer"],
        "adjs": ["conventional"],
        "locations": ["meats"],
        "properties": ["closed"],
        "desc": [None],
    },
    "basket": {
        "type": "c",
        "names": ["basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["checkout"],
        "properties": ["closed"],
        "desc": [None],
    },
    "veg basket": {
        "type": "c",
        "names": ["veg basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["vegetables"],
        "properties": ["closed"],
        "desc": [None],
    },
    "fruit basket": {
        "type": "c",
        "names": ["fruit basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["fruits"],
        "properties": ["closed"],
        "desc": [None],
    },
    "red veg basket": {
        "type": "c",
        "names": ["red veg basket", "red vegetable basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["vegetables"],
        "properties": ["closed"],
        "desc": [None],
    },
    "red fruit basket": {
        "type": "c",
        "names": ["red fruit basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["fruits"],
        "properties": ["closed"],
        "desc": [None],
    },
    "green veg basket": {
        "type": "c",
        "names": ["green veg basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["vegetables"],
        "properties": ["closed"],
        "desc": [None],
    },
    "green fruit basket": {
        "type": "c",
        "names": ["green fruit basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["fruits"],
        "properties": ["closed"],
        "desc": [None],
    },
    "yellow basket": {
        "type": "c",
        "names": ["yellow basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["fruits"],
        "properties": ["closed"],
        "desc": [None],
    },
    "orange basket": {
        "type": "c",
        "names": ["orange basket"],
        "adjs": ["woven", "traditional", "cute"],
        "locations": ["vegetables"],
        "properties": ["closed"],
        "desc": [None],
    },
    "counter": {
        "type": "s",
        "names": ["counter", "checkout counter"],
        "adjs": ["high-tech", "vast"],
        "locations": ["checkout"],
        "properties": [],
        "desc": [None],
    },
    "stove": {
        "type": "stove",
        "names": ["stove"],
        "adjs": ["conventional"],
        "locations": ["checkout"],
        "properties": [],
        "desc": ["Useful for frying things."],
    },
    "oven": {
        "type": "oven",
        "names": ["oven"],
        "adjs": ["conventional"],
        "locations": ["checkout"],
        "properties": [],
        "desc": ["Useful for roasting things."],
    },

    # Pantry
    "shelf": {
        "type": "s",
        "names": ["shelf"],
        "adjs": ["wooden"],
        "locations": ["checkout"],
        "properties": [],
        "desc": [None],
    },
    "hygiene shelf": {
        "type": "s",
        "names": ["hygiene shelf"],
        "adjs": ["wooden"],
        "locations": ["hygiene"],
        "properties": [],
        "desc": [None],
    },
    "ingredients shelf": {
        "type": "s",
        "names": ["ingredients shelf"],
        "adjs": ["wooden"],
        "locations": ["ingredients"],
        "properties": [],
        "desc": [None],
    },
    "stationery shelf": {
        "type": "s",
        "names": ["stationery shelf"],
        "adjs": ["wooden"],
        "locations": ["stationery"],
        "properties": [],
        "desc": [None],
    },
    
    "hardware shelf": {
        "type": "s",
        "names": ["hardware shelf", "tech shelf"],
        "adjs": ["wooden"],
        "locations": ["hardware"],
        "properties": [],
        "desc": [None],
    },

    # Backyard
    "BBQ": {
        "type": "toaster",
        "names": ["BBQ"],
        "adjs": ["recent"],
        "locations": ["restaurant"],
        "properties": [],
        "desc": ["Useful for grilling things."],
    },
    "restaurant table": {
        "type": "s",
        "names": ["restaurant table"],
        "adjs": ["stylish", "chic"],
        "locations": ["restaurant"],
        "properties": [],
        "desc": [None],
    },
    "restaurant chair": {
        "type": "s",
        "names": ["restaurant chair"],
        "adjs": ["stylish", "chic"],
        "locations": ["restaurant"],
        "properties": [],
        "desc": [None],
    },

    # Supermarket
    "showcase": {
        "type": "s",
        "names": ["showcase"],
        "adjs": ["metallic"],
        "locations": ["checkout"],
        "properties": [],
        "desc": [None],
    },
    "hygiene showcase": {
        "type": "s",
        "names": ["hygiene showcase"],
        "adjs": ["metallic"],
        "locations": ["hygiene"],
        "properties": [],
        "desc": [None],
    },
    "clothing showcase": {
        "type": "s",
        "names": ["clothing showcase"],
        "adjs": ["metallic"],
        "locations": ["clothing"],
        "properties": [],
        "desc": [None],
    },
    "meats showcase": {
        "type": "s",
        "names": ["meats showcase"],
        "adjs": ["metallic"],
        "locations": ["meats"],
        "properties": [],
        "desc": [None],
    },
    "ingreditents showcase": {
        "type": "s",
        "names": ["ingredients showcase"],
        "adjs": ["metallic"],
        "locations": ["ingredients"],
        "properties": [],
        "desc": [None],
    },
    # Livingroom
    "sofa": {
        "type": "s",
        "names": ["sofa", "couch"],
        "adjs": ["comfy"],
        "locations": ["restaurant"],
        "properties": [],
        "desc": [None],
    },

    # Bathroom
    "toilet": {
        "type": "s",
        "names": ["toilet"],
        "adjs": ["white"],
        "locations": ["restaurant"],
        "properties": [],
        "desc": [None],
    },
    # "bath": {
    #     "type": "unclosable-container",
    #     "names": ["bathtub"],
    #     "adjs": ["white"],
    #     "locations": ["bathroom"],
    #     "properties": [],
    #     "desc": [None],
    # },

    # Shed
    "donation box": {
        "type": "c",
        "names": ["donation box", "charity box"],
        "adjs": ["transparent"],
        "locations": ["checkout", "restaurant"],
        "properties": ["closed"],
        "desc": [None],
    },
    "tips box": {
        "type": "c",
        "names": ["tips box", "tip box"],
        "adjs": ["transparent"],
        "locations": ["restaurant"],
        "properties": ["closed"],
        "desc": [None],
    },

}

NEIGHBORS = {
    "checkout": ["fruits", "stationery", "meats", "vegetables"],
    "vegetables": ["checkout"],
    "fruits": ["checkout", "clothing", "restaurant", "meats"],
    "ingredients": ["meats"],
    "clothing": ["fruits", "meats"],
    "stationery": ["checkout", "hardware", "hygiene", "meats"],
    "hardware": ["stationery"],
    "hygiene": ["stationery"],
    "restaurant": ["fruits", "security", "meats"],
    "security": ["driveway", "carparking"],
    "meats": ["fruits", "checkout", "clothing", "ingredients", "restaurant", "stationery"],
    "carparking": ["security"],
}

ROOMS = [
    ["checkout"],
    ["vegetables", "fruits", "meats", "clothing", "ingredients"],
    ["hygiene", "hardware", "stationery", "restaurant"],
    ["security", "carparking"]
]

DOORS = [
    {
        "path": ("vegetable", "checkout"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
    {
        "path": ("checkout", "stationery"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
    {
        "path": ("meats", "stationery"),
        "names": ["sliding patio door", "patio door", "screen door"],
    },
    {
        "path": ("stationery", "hygiene"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
    {
        "path": ("fruits", "restaurant"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
    {
        "path": ("meats", "restaurant"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
    {
        "path": ("carparking", "security"),
        "names": ["sliding glass door", "plain door", "screen door"],
    },
]

def pick_name(M, names, rng):
    names = list(names)
    rng.shuffle(names)
    for name in names:
        if M.find_by_name(name) is None:
            return name

    assert False
    return None


def get_food_preparations(foods):
    food_preparations = {}
    for f in foods:
        v = FOODS[f]
        cookings = ["raw"]
        if "cookable" in v["properties"]:
            cookings = ["grilled", "fried", "roasted"]
            if "needs_cooking" not in v["properties"]:
                cookings.append("raw")

        cuttings = ["uncut"]
        if "cuttable" in v["properties"]:
            cuttings = ["uncut", "chopped", "sliced", "diced"]

        food_preparations[f] = list(itertools.product(cookings, cuttings))

    return food_preparations


def pick_location(M, locations, rng):
    locations = list(locations)
    rng.shuffle(locations)
    for location in locations:
        holder_name = location.split(".")[-1]
        holder = M.find_by_name(holder_name)
        if holder:
            return holder

    return None


def place_food(M, name, rng, place_it=True):
    holder = pick_location(M, FOODS[name]["locations"], rng)
    if holder is None and place_it:
        return None

    food = M.new(type=FOODS[name].get("type", "f"), name=name)
    food.infos.adj = ""
    food.infos.noun = name
    if "indefinite" in FOODS[name]:
        food.infos.indefinite = FOODS[name]["indefinite"]

    for property_ in FOODS[name]["properties"]:
        food.add_property(property_)

    if place_it:
        holder.add(food)

    return food


def place_foods(M, foods, rng):
    entities = []
    for name in foods:
        food = place_food(M, name, rng)
        if food:
            entities.append(food)

    return entities


def place_random_foods(M, nb_foods, rng, allowed_foods=FOODS):
    seen = set(food.name for food in M.findall(type="f"))
    foods = [name for name in allowed_foods if name not in seen]
    rng.shuffle(foods)
    entities = []
    for food in foods:
        if len(entities) >= nb_foods:
            break

        entities += place_foods(M, [food], rng)

    return entities


def place_entity(M, name, rng) -> WorldEntity:
    holder = pick_location(M, ENTITIES[name]["locations"], rng)
    if holder is None:
        return None  # Nowhere to place it.

    entity = M.new(type=ENTITIES[name]["type"], name=name)
    entity.infos.adj = ENTITIES[name]["adjs"][0]
    entity.infos.noun = name
    entity.infos.desc = ENTITIES[name]["desc"][0]
    for property_ in ENTITIES[name]["properties"]:
        entity.add_property(property_)

    holder.add(entity)
    return entity


def place_entities(M, names, rng):
    return [place_entity(M, name, rng) for name in names]


def place_random_furnitures(M, nb_furnitures, rng):
    furnitures = [k for k, v in ENTITIES.items() if v["type"] not in ["o", "f"]]
    # Skip existing furnitures.
    furnitures = [furniture for furniture in furnitures if not M.find_by_name(furniture)]
    rng.shuffle(furnitures)
    return place_entities(M, furnitures[:nb_furnitures], rng)


def move(M, G, start, end):
    path = nx.algorithms.shortest_path(G, start.id, end.id)
    commands = []
    current_room = start
    for node in path[1:]:
        previous_room = current_room
        direction, current_room = [(exit.direction, exit.dest.src) for exit in previous_room.exits.values()
                                   if exit.dest and exit.dest.src.id == node][0]

        path = M.find_path(previous_room, current_room)
        if path.door:
            commands.append("open {}".format(path.door.name))

        commands.append("go {}".format(direction))

    return commands


def compute_graph(M):
    G = nx.Graph()
    constraints = []
    G.add_nodes_from(room.id for room in M.rooms)

    def is_positioning_fact(proposition: Proposition):
        return proposition.name in ["north_of", "south_of", "east_of", "west_of"]

    positioning_facts = [fact for fact in M.facts if is_positioning_fact(fact)]
    for fact in positioning_facts:
        G.add_edge(fact.arguments[0].name, fact.arguments[1].name)
        constraints.append((fact.arguments[0].name, fact.name[:-3], fact.arguments[1].name))

    return G


class RandomWalk:
    def __init__(self, neighbors, size=(5, 5), max_attempts=200, rng=None):
        self.max_attempts = max_attempts
        self.neighbors = neighbors
        self.rng = rng or np.random.RandomState(1234)
        self.grid = nx.grid_2d_graph(size[0], size[1], create_using=nx.OrderedGraph())
        self.nb_attempts = 0

    def _walk(self, G, node, remaining):
        if len(remaining) == 0:
            return G

        self.nb_attempts += 1
        if self.nb_attempts > self.max_attempts:
            return None

        nodes = list(self.grid[node])
        self.rng.shuffle(nodes)
        for node_ in nodes:
            neighbors = self.neighbors[G.nodes[node]["name"]]
            if node_ in G:
                if G.nodes[node_]["name"] not in neighbors:
                    continue

                new_G = G.copy()
                new_G.add_edge(node, node_, has_door=False, door_state=None, door_name=None)
                new_G = self._walk(new_G, node_, remaining)
                if new_G:
                    return new_G

            else:
                neighbors = [n for n in neighbors if n in remaining]
                self.rng.shuffle(neighbors)

                for neighbor in neighbors:
                    new_G = G.copy()
                    new_G.add_node(node_, id="r_{}".format(len(new_G)), name=neighbor)
                    new_G.add_edge(node, node_, has_door=False, door_state=None, door_name=None)
                    new_G = self._walk(new_G, node_, remaining - {neighbor})
                    if new_G:
                        return new_G

        return None

    def place_rooms(self, rooms):
        nodes = list(self.grid)
        self.rng.shuffle(nodes)

        for start in nodes:
            G = nx.OrderedGraph()
            room = rooms[0][0]
            G.add_node(start, id="r_{}".format(len(G)), name=room, start=True)

            for group in rooms:
                self.nb_attempts = 0
                G = self._walk(G, start, set(group) - {room})
                if not G:
                    break

            if G:
                return G

        return None


def make_graph_world(rng: RandomState, rooms: List[List[str]],
                     neighbors: Dict[str, List[str]], size: Tuple[int, int] = (5, 5)):
    walker = RandomWalk(neighbors=neighbors, size=(5, 5), rng=rng)
    G = walker.place_rooms(rooms)
    return G


def make(settings: Mapping[str, str], options: Optional[GameOptions] = None) -> textworld.Game:
    """ Make a Cooking game.

    Arguments:
        settings: Difficulty settings (see notes).
        options:
            For customizing the game generation (see
            :py:class:`textworld.GameOptions <textworld.generator.game.GameOptions>`
            for the list of available options).

    Returns:
        Generated game.

    Notes:
        The settings that can be provided are:

        * recipe : Number of ingredients in the recipe.
        * take : Number of ingredients to fetch. It must be less
          or equal to the value of the `recipe` skill.
        * open : Whether containers/doors need to be opened.
        * cook : Whether some ingredients need to be cooked.
        * cut : Whether some ingredients need to be cut.
        * drop : Whether the player's inventory has limited capacity.
        * go : Number of locations in the game (1, 6, or 10).
    """
    options = options or GameOptions()

    # Load knowledge base specific to this challenge.
    if settings.get("drop"):
        options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_DROP_PATH, grammar_path=KB_GRAMMAR_PATH)
    else:
        options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_PATH, grammar_path=KB_GRAMMAR_PATH)

    rngs = options.rngs
    rng_map = rngs['map']
    rng_objects = rngs['objects']
    rng_grammar = rngs['grammar']
    rng_quest = rngs['quest']
    rng_recipe = np.random.RandomState(settings["recipe_seed"])

    allowed_foods = list(FOODS)
    allowed_food_preparations = get_food_preparations(list(FOODS))
    if settings["split"] == "train":
        allowed_foods = list(FOODS_SPLITS['train'])
        allowed_food_preparations = dict(FOOD_PREPARATIONS_SPLITS['train'])
    elif settings["split"] == "valid":
        allowed_foods = list(FOODS_SPLITS['valid'])
        allowed_food_preparations = get_food_preparations(FOODS_SPLITS['valid'])
        # Also add food from the training set but with different preparations.
        allowed_foods += [f for f in FOODS if f in FOODS_SPLITS['train']]
        allowed_food_preparations.update(dict(FOOD_PREPARATIONS_SPLITS['valid']))
    elif settings["split"] == "test":
        allowed_foods = list(FOODS_SPLITS['test'])
        allowed_food_preparations = get_food_preparations(FOODS_SPLITS['test'])
        # Also add food from the training set but with different preparations.
        allowed_foods += [f for f in FOODS if f in FOODS_SPLITS['train']]
        allowed_food_preparations.update(dict(FOOD_PREPARATIONS_SPLITS['test']))

    if settings.get("cut"):
        # If "cut" skill is specified, remove all "uncut" preparations.
        for food, preparations in allowed_food_preparations.items():
            allowed_food_preparations[food] = [preparation for preparation in preparations if "uncut" not in preparation]

    if settings.get("cook"):
        # If "cook" skill is specified, remove all "raw" preparations.
        for food, preparations in list(allowed_food_preparations.items()):
            allowed_food_preparations[food] = [preparation for preparation in preparations if "raw" not in preparation]
            if len(allowed_food_preparations[food]) == 0:
                del allowed_food_preparations[food]
                allowed_foods.remove(food)

    M = textworld.GameMaker(options)

    recipe = M.new(type='RECIPE', name='')
    meal = M.new(type='meal', name='items')
    M.add_fact("out", meal, recipe)
    meal.add_property("edible")
    M.nowhere.append(recipe)  # Out of play object.
    M.nowhere.append(meal)  # Out of play object.

    options.nb_rooms = settings.get("go", 1)
    if options.nb_rooms == 1:
        rooms_to_place = ROOMS[:1]
    elif options.nb_rooms == 6:
        rooms_to_place = ROOMS[:2]
    elif options.nb_rooms == 10:
        rooms_to_place = ROOMS[:3]
    else:
        raise ValueError("Shopping games can only have {1, 6, 10} rooms.")

    G = make_graph_world(rng_map, rooms_to_place, NEIGHBORS, size=(5, 5))
    rooms = M.import_graph(G)

    # Add doors
    for infos in DOORS:
        room1 = M.find_by_name(infos["path"][0])
        room2 = M.find_by_name(infos["path"][1])
        if room1 is None or room2 is None:
            continue  # This door doesn't exist in this world.

        path = M.find_path(room1, room2)
        if path:
            assert path.door is None
            name = pick_name(M, infos["names"], rng_objects)
            door = M.new_door(path, name)
            door.add_property("closed")

    # Find kitchen.
    kitchen = M.find_by_name("checkout") # Changed to checkout

    # The following predicates will be used to force the "prepare meal"
    # command to happen in the kitchen.
    M.add_fact("cooking_location", kitchen, recipe)

    # Place some default furnitures.
    place_entities(M, ["basket", "counter", "fridge", "shelf", "showcase"], rng_objects)

    # Place some random furnitures.
    nb_furnitures = rng_objects.randint(len(rooms), len(ENTITIES) + 1)
    place_random_furnitures(M, nb_furnitures, rng_objects)

    # Place the cookbook and knife somewhere.
    cookbook = place_entity(M, "shopping list", rng_objects) # shopping list instead of cookbook
    cookbook.infos.synonyms = ["list"]
    if rng_objects.rand() > 0.5 or settings.get("cut"):
        knife = place_entity(M, "knife", rng_objects)

    start_room = rng_map.choice(M.rooms)
    M.set_player(start_room)

    M.grammar = textworld.generator.make_grammar(options.grammar, rng=rng_grammar)
    
    '''
    # Remove every food preparation with grilled, if there is no BBQ.
    if M.find_by_name("BBQ") is None:
        for name, food_preparations in allowed_food_preparations.items():
            allowed_food_preparations[name] = [food_preparation for food_preparation in food_preparations
                                               if "grilled" not in food_preparation]

        # Disallow food with an empty preparation list.
        allowed_foods = [name for name in allowed_foods if allowed_food_preparations[name]]
    '''
    # Decide which ingredients are needed.
    nb_ingredients = settings.get("recipe", 1)
    assert nb_ingredients > 0 and nb_ingredients <= 5, "recipe must have {1,2,3,4,5} ingredients."
    ingredient_foods = place_random_foods(M, nb_ingredients, rng_quest, allowed_foods)

    # Sort by name (to help differentiate unique recipes).
    ingredient_foods = sorted(ingredient_foods, key=lambda f: f.name)

    # Decide on how the ingredients should be processed.
    ingredients = []
    for i, food in enumerate(ingredient_foods):
        food_preparations = allowed_food_preparations[food.name]
        idx = rng_quest.randint(0, len(food_preparations))
        type_of_cooking, type_of_cutting = food_preparations[idx]
        ingredients.append((food, type_of_cooking, type_of_cutting))

        # ingredient = M.new(type="ingredient", name="")
        # food.add_property("ingredient_{}".format(i + 1))
        # M.add_fact("base", food, ingredient)
        # M.add_fact(type_of_cutting, ingredient)
        # M.add_fact(type_of_cooking, ingredient)
        # M.add_fact("in", ingredient, recipe)
        # M.nowhere.append(ingredient)

    # Move ingredients in the player's inventory according to the `take` skill.
    #nb_ingredients_already_in_inventory = nb_ingredients - settings.get("take", 0)
    #shuffled_ingredients = list(ingredient_foods)
    #rng_quest.shuffle(shuffled_ingredients)
    #for ingredient in shuffled_ingredients[:nb_ingredients_already_in_inventory]:
        #M.move(ingredient, M.inventory)

    # Compute inventory capacity.
    inventory_limit = 10  # More than enough.
    if settings.get("drop"):
        inventory_limit = nb_ingredients
        if nb_ingredients == 1 and settings.get("cut"):
            inventory_limit += 1  # So we can hold the knife along with the ingredient.

    # Add distractors for each ingredient.
    def _place_one_distractor(candidates, ingredient):
        rng_objects.shuffle(candidates)
        for food_name in candidates:
            distractor = M.find_by_name(food_name)
            if distractor:
                if distractor.parent == ingredient.parent:
                    break  # That object already exists and is considered as a distractor.

                continue  # That object already exists. Can't used it as distractor.

            # Place the distractor in the same "container" as the ingredient.
            distractor = place_food(M, food_name, rng_objects, place_it=False)
            ingredient.parent.add(distractor)
            break

    for ingredient in ingredient_foods:
        if ingredient.parent == M.inventory and nb_ingredients_already_in_inventory >= inventory_limit:
            # If ingredient is in the inventory but inventory is full, do not add distractors.
            continue

        splits = ingredient.name.split()
        if len(splits) == 1:
            continue  # No distractors.

        prefix, suffix = splits[0], splits[-1]
        same_prefix_list = [f for f in allowed_foods if f.startswith(prefix) if f != ingredient.name]
        same_suffix_list = [f for f in allowed_foods if f.endswith(suffix) if f != ingredient.name]

        if same_prefix_list:
            _place_one_distractor(same_prefix_list, ingredient)

        if same_suffix_list:
            _place_one_distractor(same_suffix_list, ingredient)

    # Add distractors foods. The amount is drawn from N(nb_ingredients, 3).
    nb_distractors = abs(int(rng_objects.randn(1) * 3 + nb_ingredients))
    distractors = place_random_foods(M, nb_distractors, rng_objects, allowed_foods)

    # If recipe_seed is positive, a new recipe is sampled.
    if settings["recipe_seed"] > 0:
        assert settings.get("take", 0), "Shuffle recipe requires the 'take' skill."
        potential_ingredients = ingredient_foods + distractors
        rng_recipe.shuffle(potential_ingredients)
        ingredient_foods = potential_ingredients[:nb_ingredients]

        # Decide on how the ingredients of the new recipe should be processed.
        ingredients = []
        for i, food in enumerate(ingredient_foods):
            food_preparations = allowed_food_preparations[food.name]
            idx = rng_recipe.randint(0, len(food_preparations))
            type_of_cooking, type_of_cutting = food_preparations[idx]
            ingredients.append((food, type_of_cooking, type_of_cutting))

    # Add necessary facts about the recipe.
    for i, (food, type_of_cooking, type_of_cutting) in enumerate(ingredients):
        ingredient = M.new(type="ingredient", name="")
        food.add_property("ingredient_{}".format(i + 1))
        M.add_fact("base", food, ingredient)
        M.add_fact(type_of_cutting, ingredient)
        M.add_fact(type_of_cooking, ingredient)
        M.add_fact("in", ingredient, recipe)
        M.nowhere.append(ingredient)

    # Depending on the skills and how the ingredient should be processed
    # we change the predicates of the food objects accordingly.
    for food, type_of_cooking, type_of_cutting in ingredients:
        if not settings.get("cook"):  # Food should already be cooked accordingly.
            food.add_property(type_of_cooking)
            food.add_property("cooked")
            if food.has_property("inedible"):
                food.add_property("edible")
                food.remove_property("inedible")
            if food.has_property("raw"):
                food.remove_property("raw")
            if food.has_property("needs_cooking"):
                food.remove_property("needs_cooking")

        if not settings.get("cut"):  # Food should already be cut accordingly.
            food.add_property(type_of_cutting)
            food.remove_property("uncut")

    if not settings.get("open"):
        for entity in M._entities.values():
            if entity.has_property("closed"):
                entity.remove_property("closed")
                entity.add_property("open")

    walkthrough = []
    # Build TextWorld quests.
    quests = []
    consumed_ingredient_events = []
    for i, ingredient in enumerate(ingredients):
        ingredient_consumed = Event(conditions={M.new_fact("consumed", ingredient[0])})
        consumed_ingredient_events.append(ingredient_consumed)
        ingredient_burned = Event(conditions={M.new_fact("burned", ingredient[0])})
        quests.append(Quest(win_events=[], fail_events=[ingredient_burned]))

        if ingredient[0] not in M.inventory:
            holding_ingredient = Event(conditions={M.new_fact("in", ingredient[0], M.inventory)})
            quests.append(Quest(win_events=[holding_ingredient]))

        win_events = []
        if ingredient[1] != TYPES_OF_COOKING[0] and not ingredient[0].has_property(ingredient[1]):
            win_events += [Event(conditions={M.new_fact(ingredient[1], ingredient[0])})]

        fail_events = [Event(conditions={M.new_fact(t, ingredient[0])})
                       for t in set(TYPES_OF_COOKING[1:]) - {ingredient[1]}]  # Wrong cooking.

        quests.append(Quest(win_events=win_events, fail_events=[ingredient_consumed] + fail_events))

        win_events = []
        if ingredient[2] != TYPES_OF_CUTTING[0] and not ingredient[0].has_property(ingredient[2]):
            win_events += [Event(conditions={M.new_fact(ingredient[2], ingredient[0])})]

        fail_events = [Event(conditions={M.new_fact(t, ingredient[0])})
                       for t in set(TYPES_OF_CUTTING[1:]) - {ingredient[2]}]  # Wrong cutting.

        quests.append(Quest(win_events=win_events, fail_events=[ingredient_consumed] + fail_events))
    
    
        
    #quests.append(Quest(win_events=[itemsCounter], fail_events=consumed_ingredient_events))
    #holding_meal = Event(conditions={M.new_fact("in", meal, M.inventory)})
    #quests.append(Quest(win_events=[holding_meal], fail_events=consumed_ingredient_events))

    #meal_burned = Event(conditions={M.new_fact("burned", meal)})
    #meal_consumed = Event(conditions={M.new_fact("consumed", meal)})
    #quests.append(Quest(win_events=[meal_consumed], fail_events=[meal_burned]))

    M.quests = quests

    G = compute_graph(M)  # Needed by the move(...) function called below.

    # Build walkthrough.
    current_room = start_room
    walkthrough = []

    # Start by checking the inventory.
    #walkthrough.append("inventory")

    # 0. Find the kitchen and read the cookbook.
    walkthrough += move(M, G, current_room, kitchen)
    current_room = kitchen
    walkthrough.append("examine shopping list")

    # 1. Drop unneeded objects.
    for entity in M.inventory.content:
        if entity not in ingredient_foods:
            walkthrough.append("drop {}".format(entity.name))

    # 2. Collect the ingredients.
    for food, type_of_cooking, type_of_cutting in ingredients:
        if food.parent == M.inventory:
            continue

        food_room = food.parent.parent if food.parent.parent else food.parent
        walkthrough += move(M, G, current_room, food_room)

        if food.parent.has_property("closed"):
            walkthrough.append("open {}".format(food.parent.name))

        if food.parent == food_room:
            walkthrough.append("take {}".format(food.name))
        else:
            walkthrough.append("take {} from {}".format(food.name, food.parent.name))

        current_room = food_room

    # 3. Go back to the kitchen.
    #walkthrough += move(M, G, current_room, kitchen)
    
    '''
    # 4. Process ingredients (cook).
    if settings.get("cook"):
        for food, type_of_cooking, _ in ingredients:
            if type_of_cooking == "fried":
                stove = M.find_by_name("stove")
                walkthrough.append("cook {} with {}".format(food.name, stove.name))
            elif type_of_cooking == "roasted":
                oven = M.find_by_name("oven")
                walkthrough.append("cook {} with {}".format(food.name, oven.name))
            elif type_of_cooking == "grilled":
                toaster = M.find_by_name("BBQ")
                # 3.a move to the backyard.
                walkthrough += move(M, G, kitchen, toaster.parent)
                # 3.b grill the food.
                walkthrough.append("cook {} with {}".format(food.name, toaster.name))
                # 3.c move back to the kitchen.
                walkthrough += move(M, G, toaster.parent, kitchen)

    # 5. Process ingredients (cut).
    if settings.get("cut"):
        free_up_space = settings.get("drop") and not len(ingredients) == 1
        knife = M.find_by_name("knife")
        if knife:
            knife_location = knife.parent.name
            knife_on_the_floor = knife_location == "checkout" # changed from kitchen
            for i, (food, _, type_of_cutting) in enumerate(ingredients):
                if type_of_cutting == "uncut":
                    continue

                if free_up_space:
                    ingredient_to_drop = ingredients[(i + 1) % len(ingredients)][0]
                    walkthrough.append("drop {}".format(ingredient_to_drop.name))

                # Assume knife is reachable.
                if knife_on_the_floor:
                    walkthrough.append("take {}".format(knife.name))
                else:
                    walkthrough.append("take {} from {}".format(knife.name, knife_location))

                if type_of_cutting == "chopped":
                    walkthrough.append("chop {} with {}".format(food.name, knife.name))
                elif type_of_cutting == "sliced":
                    walkthrough.append("slice {} with {}".format(food.name, knife.name))
                elif type_of_cutting == "diced":
                    walkthrough.append("dice {} with {}".format(food.name, knife.name))

                walkthrough.append("drop {}".format(knife.name))
                knife_on_the_floor = True
                if free_up_space:
                    walkthrough.append("take {}".format(ingredient_to_drop.name))
    '''
    # 6. Prepare and eat meal.
    #walkthrough.append("prepare meal")
    #walkthrough.append("eat items")

    cookbook_desc = "You examine the shopping list and start reading:\n"
    recipe = textwrap.dedent(
        """
        Shopping list #1
        ---------
        Gather all following items and follow the directions to enjoy a wonderful and hassle-free day of shopping.

        Items:
        {ingredients}

        Directions:
        {directions}
        """
    )
    recipe_ingredients = "\n".join(ingredient[0].name for ingredient in ingredients)
    
    recipe_directions = []
    '''
    for ingredient in ingredients:
        cutting_verb = TYPES_OF_CUTTING_VERBS.get(ingredient[2])
        if cutting_verb:
            recipe_directions.append(cutting_verb + " the " + ingredient[0].name)

        cooking_verb = TYPES_OF_COOKING_VERBS.get(ingredient[1])
        if cooking_verb:
            recipe_directions.append(cooking_verb + " the " + ingredient[0].name)
    '''
    recipe_directions.append("Gather all above items and enjoy!")
    recipe_directions = "\n  ".join(recipe_directions)
    recipe = recipe.format(ingredients=recipe_ingredients, directions=recipe_directions)
    cookbook.infos.desc = cookbook_desc + recipe

    if settings.get("drop"):
        # Limit capacity of the inventory.
        for i in range(inventory_limit):
            slot = M.new(type="slot", name="")
            if i < len(M.inventory.content):
                slot.add_property("used")
            else:
                slot.add_property("free")

            M.nowhere.append(slot)

    # Sanity checks:
    for entity in M._entities.values():
        if entity.type in ["c", "d"]:
            if not (entity.has_property("closed")
                    or entity.has_property("open")
                    or entity.has_property("locked")):
                raise ValueError("Forgot to add closed/locked/open property for '{}'.".format(entity.name))

    if not settings.get("drop"):
        M.set_walkthrough(walkthrough)
    else:
        pass  # BUG: With `--drop` having several "slots" causes issues with dependency tree.

    game = M.build()

    # Collect infos about this game.
    metadata = {
        "seeds": options.seeds,
        "goal": cookbook.infos.desc,
        "recipe": recipe,
        "ingredients": [(food.name, cooking, cutting) for food, cooking, cutting in ingredients],
        "settings": settings,
        "entities": [e.name for e in M._entities.values() if e.name],
        "nb_distractors": nb_distractors,
        "walkthrough": walkthrough,
        "max_score": sum(quest.reward for quest in game.quests),
    }

    objective = ("You are eager to shop! Let's get you sorted to quench your desire. Check the shopping list"
                 " in the checkout for the items needed and directions. Once done, enjoy your items!")
    game.objective = objective

    game.metadata = metadata
    skills_uuid = "+".join("{}{}".format(k, "" if settings[k] is True else settings[k])
                           for k in SKILLS if k in settings and settings[k])
    uuid = "tw-testing{split}-{specs}-{seeds}"
    uuid = uuid.format(split="-{}".format(settings["split"]) if settings.get("split") else "",
                       specs=skills_uuid,
                       seeds=encode_seeds([options.seeds[k] for k in sorted(options.seeds)]))
    game.metadata["uuid"] = uuid
    return game


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()

    group = parser.add_argument_group('The Shopping Game settings')
    group.add_argument("--recipe", type=int, default=1, metavar="INT",
                       help="Number of items in the shopping list. Default: %(default)s")
    group.add_argument("--take", type=int, default=0, metavar="INT",
                       help="Number of items to find. It must be less or equal to"
                            " the value of `--recipe`. Default: %(default)s")
    group.add_argument("--go", type=int, default=1, choices=[1, 6, 10],
                       help="Number of locations in the game (1, 6 or 10). Default: %(default)s")
    group.add_argument('--open', action="store_true",
                       help="Whether containers/doors need to be opened.")
    group.add_argument('--cook', action="store_true",
                       help="Whether some ingredients need to be cooked.")
    group.add_argument('--cut', action="store_true",
                       help="Whether some ingredients need to be cut.")
    group.add_argument('--drop', action="store_true",
                       help="Whether the player's inventory has limited capacity.")
    group.add_argument("--recipe-seed", type=int, default=0, metavar="INT",
                       help="Random seed used for generating the recipe. Default: %(default)s")

    group.add_argument("--split", choices=["train", "valid", "test"],
                       help="Specify the game distribution to use. Food items (adj-noun pairs) are split in three subsets."
                            " Also, the way the training food items can be prepared is further divided in three subsets.\n\n"
                            "* train: training food and their corresponding training preparations\n"
                            "* valid: valid food + training food but with unseen valid preparations\n"
                            "* test: test food + training food but with unseen test preparations\n\n"
                            " Default: game is drawn from the joint distribution over train, valid, and test.")

    return parser


register(name="tw-testing",
         desc=("Generate shopping games similar to the cooking games used for the"
               " First TextWorld Problem (FTWP) competition (https://aka.ms/ftwp)."),
         make=make,
         add_arguments=build_argparser)
