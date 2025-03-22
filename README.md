
# quick instructions
To run this code, go to config.py, set the options you want, go to run.sh, set the options you want, and then call "sh run.sh". 


# more info

This is some code written by Mark. 

Main point of repo is to help Courant people start small projects.

The code captures many small details / tips from collaborators (Valentin de Bortoli, Raghav Singhal, Michael Albergo, Nick Boffi, Jiaxin Shi, Will Grathwohl, Willis Ma, Saining Xie, etc)

# the loss function

This code supports velocity prediction, noise prediction, score prediction etc.... We distinguish between what your model outputs versus which squared error you compute. For example, you could output a noise prediction but compute a score loss, or output a velocity and compute a noise loss. Or just output a velocity and compute a velocity loss.
```
# This is whatever your model approximates
model_out = model_fn(xt, t, label)

# This an object containing all possible prediction types
model_obj = self.model_convert_fn(model_out, ...)

# This is your estimate of the target (a conversion of whatever your model outputted)
model_pred = getattr(model_obj, target_type)

# object containing all possible targets
target_fn = prediction.get_target_fn()
target_obj = target_fn(t, xt, x0, x1, ...)

# specific target for squared loss
target = getattr(target_obj, target_type)

# the squared error loss
sq_err = utils.image_square_error(model_pred, target)
```

# Augmentation

Note that we don't use the pytorch + torchvision data augmentation in the dataloader. We don't give the dataloader any augmentations and instead define an augmenter ourselves. The augmenter is defined in augmentations.py and used in the prepare batch function. Extend or replace this as needed, perhaps with the torchvision augmentations.

# Customizing this code

Still experimenting with the design, but I think for now it's like this:
- in the trainer file, there is a BaseDiffusionTrainer
- this contains a prepare_batch_fn (which preproceses the batch) and a loss_fn (which computes the loss
- the BaseDiffusionTrainer doesn't implement these, and it is sub-classed with an ExampleDiffusionTrainer that implements these
- you can mostly edit the ExampleDiffusionTrainer to do your own data preprocessing and loss function computation
- besides that you can add a new dataset in data_utils.setup_data_train_and_test
- and add a new model architecture

# Files and stuff

Files:

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
- **saving.py**:
	- logic for checkpointing and restoring model


Subdirectories:
- "**data**" subdirectory: this might not be present in your copy of the repo, make one. This is where you will store data.
- "**ckpts**" subdirectory: this might not be present in your copy of the repo, make one. This is where you will store model checkpoints.






