#!/bin/bash
# author: Pranay Methuku

# Running the same sequence of command line instructions every time 
rm -f detection.db
cat create_database.sql populate_categories.sql | sqlite3 detection.db
