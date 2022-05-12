# GamerBotAI

The GamerBotAI learns and plays game itself.

# Idea

We will sense the environment by screenshots and listen to the audio output from the speaker. Actions such as keyboard input,
mouse keys and cursor positions will be recorded and paired with a screenshot. A new state (new screenshot) will also be 
a valid data that will be paired with an idle action.

States and actions are captured in real time with a limited rate and all data will be sent through the pipeline to the
pre-processor. The pre-processor will then return the precessed and normalised data to another pipeline that is connected to
the agent. The agent is connected to the neural network that will be trained and will be able to play the game.

Basically we play the game for some amount of time as human for supervised learning first, and then the game
will be played by the AI itself with reinforcement learning.

# Plans
Currently, this is a work in progress, and obviously, it does not work. You can test the current working part of code with the entry
available in src.Model.data_pipeline.py.
If you have any idea please leave some comment, and we can work together. Cheers

#Antivirus
Windows security is such a pain. If any file is missing, check the antivirus.

# Windows
If you have problems installing the pyaudio, use the following command:

*pip install pipwin*

*pipwin install pyaudio*