import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        with open(filename + ".pickle", "rb") as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        # Save as a pickle file
        with open(filename + ".pickle", "wb") as f:
            pickle.dump(self.q, f)
        
        # Save as a CSV file
        with open(filename + ".csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in self.q.items():
                writer.writerow([key, value])
        
        print("Saved Q-values to file: {}.pickle and {}.csv".format(filename, filename))

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        # Exploration: Choose a random action with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            if return_q:
                return action, self.getQ(state, action)
            return action

        # Exploitation: Choose the action with the highest Q-value
        q_values = [self.getQ(state, action) for action in self.actions]
        maxQ = max(q_values)

        # Handle the case where multiple actions have the same max Q-value
        best_actions = [action for action, q in zip(self.actions, q_values) if q == maxQ]
        action = random.choice(best_actions)

        if return_q:
            return action, maxQ

        return action

        # return self.actions[1]

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
         # Get the current Q-value for (state1, action1)
        current_q_value = self.getQ(state1, action1)
    
        # Find the maximum Q-value for state2 across all possible actions
        future_q_values = [self.getQ(state2, action) for action in self.actions]
        max_future_q_value = max(future_q_values) if future_q_values else 0.0
    
        # Calculate the new Q-value using the Bellman equation
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * max_future_q_value - current_q_value)
    
        # Update the Q-table
        self.q[(state1, action1)] = new_q_value

        # self.q[(state1,action1)] = reward
