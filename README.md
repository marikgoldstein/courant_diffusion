
# quick instructions
To run this code, go to config.py, set the options you want, go to run.sh, set the options you want, and then call "sh run.sh". 


# more info

This is some code written by Mark. 

Main point of repo is to help Courant + Courant adjacent people start small projects.

The code captures many small details / tips from collaborators (Valentin de Bortoli, Michael Albergo, Nick Boffi, etc)

I purposely took a few things out relative to my heavier code bases to make code cleaner/lighter but should put the features back in. Main one for now is saving/checkpointing/reloading the model.

Finally, this code is still not "light" since it does provide some stuff like multi GPU. If there's interest in a Much Lighter code base with far fewer options, I can put that together.

I imagine that the way you might use this code is to make your own copy in a new repo rather than fork it and submit changes. Meanwhile I'll update it some more in the near future. Once the code settles, feel free to submit pull requests. Until then maybe just let me know any suggestions in person or ask me to give you pull/push access.

Main TODOs:
- add model saving code back in 
- add example of non cifar dataset
- maybe make sure code works for feature vector type data and not just image data. Requires adding another architecture.


Files:

I prefer "flat" python directories that you dont need to setup/install rather than nested ones, so I try to keep few files total. 

- **README.md**: 
    - this file
- **config.py**: 
    - all the settings for defining an experiment
- **run.sh**: 
    - calls the the main python script. 
    - This file is specific to "DDP" which is pytorch's way to run multi-gpu experiments. 
    - I use this method of launching even when there is just 1 GPU.
- **main.py**: 
    - the file that is called by run.sh. 
    - This file has a few command line arguments, and merges them with the other settings in config.py. 
    - This file sets up all the devices / dataloaders etc.
    - This file then instantiates the DiffusionTrainer object and then starts the experiment.
- **trainer.py**:
    - this is where the bulk of the training logic is, including the main loop that runs during training
- **processes.py**
    - this is where you pick a specific "process"
    - specifies alpha and sigma such that xt = alpha(t) x0 + sigma(t) x1
- **sampling.py**:
    - how to sample from your model
- **prediction.py**:
    - helps convert between scores, velocities, noise, etc. Lots of credit to Valentin de Bortoli
- **time_samplers.py**:
    - defines a python class for how to sample times in [0,1] for training the model
    - sometimes there are reasons to use things besides uniform distribution
- **unet.py**:
    - this is the architecture used for image experiments
    - it is a "simple" one as far as image networks go
    - you should replace it if you want "serious" image results
- **data_utils.py**:
    - this sets up the dataset and dataloader.
    - for now it just contains CIFAR as an example.
    - TODO I'll add one more example of how to add your own simple dataset
    - some of the stuff in here is specific to running experiments with Pytorch DDP
- **optimizers.py**:
    - Adam optimizer, learning rate scheduling, etc
- **ema.py**:
    - logic for keeping track of an exponential moving average of model weights
    - many people sample from this model rather than the model being trained
- **logging_utils.py**
    - some code for wandb library to monitor training progress online
- **utils.py**:
    - helper functions including making squared error work for images, etc



Subdirectories:
- "**data**" subdirectory: this might not be present in your copy of the repo, make one. This is where you will store data.
- "**ckpts**" subdirectory: this might not be present in your copy of the repo, make one. This is where you will store model checkpoints.






