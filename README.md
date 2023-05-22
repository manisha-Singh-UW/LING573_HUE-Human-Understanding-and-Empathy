# LING 573 Project: Predicting Human Empathy and Emotion

This is the main readme for the project.

## Deliverable D#4

### Prerequisites:
- Ensure that anaconda is installed at `~/anaconda3` folder

If necessary, download and install Anaconda by running the following commands:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

### To reproduce the results and scoring for the Adaptation Task:
The following steps will run the inference to produce the output files for devtest and evaltest. The evaluation scores for the devtest are also generated using these steps below.

1. Clone this GitHub repo to your local directory on Patas
1. Set executable permissions for `D4.cmd` and `D4_run.sh` by using these commands
    `chmod +x D4.cmd` and 
    `chmod +x D4_run.sh`
1. Run the shell script `D4_run.sh`

Note: The code can also be run on Condor by using `condor_submit D4.cmd`

Here are the steps to obtain the scores for the evaltest results:
1. The predictions.zip file has been created using the outputs of the evaltest. The contents of this file are according to the specifications of the WASSA 2023 shared task for the CONV subtask, as described at https://codalab.lisn.upsaclay.fr/competitions/11167#learn_the_details-submission-format
2. Please create a WASSA account and register for the competition at the link above.
3. Submit the predictions.zip file to the competition at the following location: https://codalab.lisn.upsaclay.fr/competitions/11167#participate
4. Once processed, the scores will be available on the 'Submit/View Results' section of the 'Participate' page.

EvalTest for the Primary Task from WASSA 2022:
<img width="981" alt="Codalab_D3_Leaderboard_v2" src="https://github.com/manisha-Singh-UW/LING573_HUE-Human-Understanding-and-Empathy/assets/11152321/541eb639-6a67-40e4-b5af-032fff2b8dd3">


EvalTest for the Adaptation Task from WASSA 2023:
<img width="981" alt="D4_submission_v1" src="https://github.com/manisha-Singh-UW/LING573_HUE-Human-Understanding-and-Empathy/assets/11152321/07ab0e48-0a7d-4ce4-9464-fa3e2b6c0043">


## Deliverable D#3

### Prerequisites:
- Ensure that anaconda is installed at `~/anaconda3` folder

If necessary, download and install Anaconda by running the following commands:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

### To reproduce the results and scoring:
The following steps will run the inference to produce the results files, and generate the evaluation scores for the best peforming model.

1. Clone this GitHub repo to your local directory on Patas
1. Set executable permissions for `D3.cmd` and `D3_run.sh` by using these commands
    `chmod +x D3.cmd` and 
    `chmod +x D3_run.sh`
1. Run the shell script `D3_run.sh`

Note: The code can also be run on Condor by using `condor_submit D3.cmd`

## Deliverable D#2

### Prerequisites:
- Ensure that anaconda is installed at `~/anaconda3` folder

If necessary, download and install Anaconda by running the following commands:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

### To reproduce the results and scoring:
The following steps will run the inference to produce the results files, and generate the evaluation scores for the best peforming model.

1. Clone this GitHub repo to your local directory on Patas
1. Set executable permissions for `D2.cmd` and `D2_run.sh` by using these commands
    `chmod +x D2.cmd`
    `chmod +x D2_run.sh`
1. Run the shell script `D2_run.sh`

Note: The condor file `D2.cmd` has been provided but due to issues with the condor environment as of 2023-04-23, this file has not been tested in the condor infrastructure.
