# build nlp.sif (singularity container), you need to define nlp.def first.
`singularity build nlp.sif nlp.def`


# build singularity with ignore fakeroot 
`singularity build --fakeroot nlp.sif singularity.def`

# To get the interactive shell inside the container
`singularity shell nlp.sif`

# run sbatch command (to get gpu nodes)
`sbatch run.sh`
(you get back the job ID)

# check job
`squeue -u <username>` list users jobs

`squeue -j <job_id>`

R-running
PD-pending