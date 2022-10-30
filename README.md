# Tracing The Boson
Repository containing functions for binary classification, trained to predict Higgs Boson precence from particle decay signatures.
Ioannis Bantzis (<ioannis.bantzis@epfl.ch>), Manos Chatzakis (<emmanouil.chatzakis@epfl.ch>), Maxence Hofer (<maxence.hofer@epfl.ch>)

## Repository organization
This repository consists of the following files:
* implementations.py: Contains all the implementations of the regression models implemented for this work.
* helpers.py: Contains some helper functions provided by the course
* utils.py: Utility functions, implemented to aid the implementation of the models
* run.py: Script reproducing the .csv file of our Aicrowd results.
    * Usage:
    ```shell 
    python3 run.py
    ```
* evaluation.py: Script for evaluating the regressions models through cross validation.
    * Example Usage:
    ```shell 
    python3 evaluation.py gd #cross validation for mean square gradient descent
    ```
    * To see detailed usage examples:
    ```shell 
    python3 evaluation.py usage #prints a helping prompt
    ```
* partial_training.py: Script that implements our partial training technique.
    * Usage:
    ```shell 
    python3 partial_training.py
    ```
* report.pdf: A two-page report about this work.
* .gitignore: A basic gitignore file for python projects. It ignores /dataset/ named directories, in order to be able to store the dataset locally at our root directory.
* README.md: This readme.

## Dataset
We use the CERN particle signature dataset, available at <https://www.aicrowd.com/challenges/epfl-machine-learning-higgs>

## About
This work was developed for the first project of the postgraduate course "cs433-Machine Learning" course of EPFL, during the fall semester of 2022.