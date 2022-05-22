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
