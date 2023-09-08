#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/final_project/utils/load_category_names.py
#
# PROGRAMMER: Lucio Yovanny Nogales Vera
# DATE CREATED: 7/09/2023
# REVISED DATE:
# PURPOSE: Load the category names from a JSON file
##
import json

def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name