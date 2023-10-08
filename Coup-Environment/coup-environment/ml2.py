import tensorflow as tf

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from env import coup_env_parallel

env = coup_env_parallel.parallel_env(render_mode="human")
env.reset()

model = Sequential()
model.add(Dense(10, input_shape=(1, 10), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(16, activation="softmax"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=16,
    nb_steps_warmup=20,
    target_model_update=0.01,
)

agent.compile(Adam(lr=0.001), metrics=["mae"])

agent.fit(env, nb_steps=100000, visualize=False, verbose=1)


while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()

# research shape of data for tensor flow. Currently have length 6 array of integers, 3 being a tuple, and 2 and 6 of which are arrays up to length 5. Do these need padding with 'None' actions? (16)
