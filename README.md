# GamerBotAI

The GamerBotAI learns and plays game itself.

# Idea

We need a pipeline to capture the actions and the screen at a certain rate limit, e.g., 10fps. There are basically 3 main action types we will have to deal with. The keyboard actions, mouse buttons and mouse movements. We can either record the data and train the AI offline or train the AI online with the pipeline.

Basically we play the game for some amount of time as a human for supervised learning first, and then we play the game as an AI for reinforcement learning.
