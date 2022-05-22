# GamerBotEnv

The GamerBotEnv provides an environment from real games or tasks for machine learning.

# Idea

We will sense the environment by screenshots and listen to the audio output from the speaker. Actions such as keyboard input,
mouse keys and cursor positions will be recorded and paired with a screenshot.

States and actions (actions for supervised only and can be ignord) are captured in real time with a limited rate and all data will be sent through the pipeline to the
pre-processor. The pre-processor will then return the precessed and normalised data to the agent. The agent response to the state and send the action to the controller which takes over real keyboard.

# Plans
Currently, this is a work in progress, and obviously, it does not work.

# Antivirus
Windows security is such a pain. If the actions.py file is missing, check the antivirus.

# Windows
Use Anaconda if you want to use GPU for deap learning.

If you have problems installing the pyaudio, use the following command:

*pip install pipwin*

*pipwin install pyaudio*

# How to use
Currently I'm writting controller so the agent can control the system.

Create the following folders under the root directory.

var/

var/nn_weights/

var/screenshots/

var/templates/

var/nn_weights/ for storing your game screenshots. Run the screenshot tool under tools/screenshot.py to takes screenshots while gaming. Then select screenshots
and crop them, then place the cropped images (templates) under var/templates folder for template matching. The template matching is used to detected the rewards.

The templates needs to be placed under subfolders under var/templates/. For exampple, var/templates/target_destroyed+0.8+100/ for templates to detect target destroyed.
Use the format "'name_of_the_reward'+'tm_threshold'+'reward_amount'/" to create these subfolders. Use _ to replace space. Save them as .png for better result. The program will read these folders automatically as long as the format is correct.

All data should be passed to the input_queue in the Agent class.
