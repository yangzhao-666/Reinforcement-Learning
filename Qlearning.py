'''
write a simple Q-learning player
'''
import numpy as np
import pandas as pd
import math

class Qlearning():
    def __init__(self, game, iterations = 100):
        self.game = game
        self.iterations = iterations
    def play(self, board):
        self.board = board
        actions = np.array(range(self.game.getActionSize()))[self.game.getValidMoves(self.board, 1) == 1]
        Q_learning = QlearningTable(actions)
        for i in range(self.iterations):
            curPlayer = 1
            board = self.board
            while(self.game.getGameEnded(board, curPlayer) == 0):
                state = hash(board.tobytes())
                bool_moves = self.game.getValidMoves(board, curPlayer)
                available_moves = np.array(range(self.game.getActionSize()))[bool_moves == 1]
                Q_learning.CheckState(state, available_moves)
                next_action = Q_learning.SelectAction(state)
                board, curPlayer = self.game.getNextState(board, curPlayer, next_action)
                next_state = hash(board.tobytes())
                if(self.game.getGameEnded(board, 1) == 1):
                    reward = 100
                elif(self.game.getGameEnded(board, -1) == 1):
                    reward = 100
                elif(self.game.getGameEnded(board, curPlayer) == 0 or self.game.getGameEnded(board, curPlayer) == 1e-4):
                    reward = -1
                bool_moves = self.game.getValidMoves(board, curPlayer)
                available_moves = np.array(range(self.game.getActionSize()))[bool_moves == 1]
                Q_learning.CheckState(next_state, available_moves)
                Q_learning.UpdateQtable(state, next_state, next_action, reward, bool(reward))
        state = hash(self.board.tobytes())
        return Q_learning.SelectAction(state)


'''
We use a DataFrame to store the q-table, it's easier to locate the value, meanwhile, we can also use a dictionary.
'''
class QlearningTable():
    def __init__(self, actions, learning_rate = 0.01, reward_discount = 0.9, epsilon = 0.9):
        self.actions = actions # a list of valid actions
        self.lr = learning_rate
        self.gamma = reward_discount
        self.epsilon = epsilon
        #create the Q-table
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

    def SelectAction(self, state):
        all_actions = self.getAvailableActions(state)
        #check if it satisify the epsilon strategy
        if(np.random.uniform() < self.epsilon):
            state_values = self.q_table.loc[state, ]
            next_action = np.random.choice(state_values[state_values==np.nanmax(state_values)].index)
        else:
            next_action = np.random.choice(all_actions)
        return(next_action)

    def getAvailableActions(self, state):
        available_actions = []
        for n,i in enumerate(self.q_table.loc[state,]):
            if(math.isnan(i) == False):
                available_actions.append(self.q_table.columns[n])
        return available_actions

    def UpdateQtable(self, state, next_state, action, reward, game_end):
        q = self.q_table.loc[state, action]
        #if the game is not ended, use the value of next state to update
        if(game_end == False):
            q_ = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            #if the game is ended, use the reward as the final reward
            q_ = reward
        self.q_table.loc[state, action] += self.lr * (q_ - q)

    def CheckState(self, state, available_actions):
        if(state not in self.q_table.index):
            self.q_table = self.q_table.append(
                    pd.Series(
                        [0]*len(self.actions),
                        index=self.q_table.columns,
                        name=state,
                        )
                    )
            # set unavaiable actions to None in order to select next action
            unvalid_moves = np.setdiff1d(self.q_table.columns, available_actions)
            for m in unvalid_moves:
                self.q_table.loc[state, m] = None
