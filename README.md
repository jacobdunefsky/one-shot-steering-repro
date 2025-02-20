# Code for our paper "Investigating Generalization of One-shot LLM Steering Vectors"

This repository contains code for reproducing the results of our paper ["Investigating Generalization of One-shot LLM Steering Vectors"](One_shot_steering.pdf).

## Requirements

Python 3.10 and Anaconda are required.

To create an Anaconda environment named `one_shot` with the rest of the requirements, run the command

`conda create --name one_shot --file packages.txt`

## Repository organization

The repository is organized such that the code required to reproduce the results in Section 3 of our paper is contained in the Jupyter Notebook `poser.ipynb`, the code required to reproduce the results in Section 4 is contained in `refusal.ipynb`, and the code required to reproduce the results in Section 5 is contained in `just_kidding.ipynb`. Instructions for using the other scripts present in this repository are given in the notebooks where they are used.

## Info on our standalone steering optimization repository

If you're interested in optimizing steering vectors for your own purposes, we recommend that you refer to the standalone repository [llm-steering-opt](https://github.com/jacobdunefsky/llm-steering-opt). Not only are the steering optimization functions in `llm-steering-opt` better documented, but importantly, `llm-steering-opt` will continue to be updated as we continue our research into steering vector optimization, while this current repository is primarily intended as a snapshot to enable reproduction of this specific paper. (Already, we've updated `llm-steering-opt` beyond the files present in this repository.)

## Citation

Please cite this work as

    @misc{dunefsky2025oneshot,
        title = {Investigating Generalization of One-shot LLM Steering Vectors},
        author = {Jacob Dunefsky and Arman Cohan},
        year = {2025},
        howpublished = {\url{https://github.com/jacobdunefsky/one-shot-steering-repro}},
    }

For any questions/comments/concerns, feel free to reach out to `jacob [dot] dunefsky [at] yale [dot] edu`.
