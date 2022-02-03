from DQN_package.DQNs import DDQN
from DQN_package.DQNs import DQN
import gym
import matplotlib.pyplot as plt

ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  agent = DDQN(action_dim=env.action_space.n,state_dim=env.reset().shape[0])

  for episode in range(EPISODE):
    # initialize task
    #print(agent.time_step)
    state = env.reset()
    # Train
    for step in range(STEP):
      action = agent.choose_epsilon_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      reward = -10 if done else 1
      agent.store_seq(state,action,reward,next_state,done)
      agent.train()
      # print(agent.time_step)
      state = next_state
      if done:
        break
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
          env.render()
          action = agent.direct_action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      if ave_reward >= 200:
        break

if __name__ == "__main__":
    main()