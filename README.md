# LING 573 Project: Predicting Human Empathy and Emotion

This is the main readme for the project.

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

1. Clone the repo to your local directory on Patas
1. Set executable permissions for `D2.cmd` and `D2_run.sh` by using these commands
    `chmod +x D2.cmd`
    `chmod +x D2_run.sh`
1. Run the shell script `D2_run.sh`

Note: The condor file `D2.cmd` has been provided but due to issues with the condor environment as of 2023-04-23, this file has not been tested in the condor infrastructure.
