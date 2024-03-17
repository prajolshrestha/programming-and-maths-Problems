#!/bin/bash
echo "Today is " `date`

echo -e "\nEnter the path to directory"
read the_path

echo -e "\nYour path has the following files and folders: "
ls ${HOME}/$the_path
