import numpy as np

class Agent:
    def __init__(self, model):
        self.model = model

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))

    def get_button(self, action):
        # map into button action
        # return action

    def get_reward(self):
        total_reward = 0.0
        self.env.reset_game()
        state = self.get_state()
        sequence = [state]
        done = False
        while not done:
            action = self.get_predicted_action(sequence)
            real_action = self.get_button(action)
            reward = self.env.act(real_action)
            total_reward += reward
            state = self.get_state()
            sequence = [state]
            done = self.env.game_over()
        self.rewards.append(total_reward)
        return total_reward
