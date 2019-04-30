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
                #print("play:bool_moves:", bool_moves)
                available_moves = np.array(range(self.game.getActionSize()))[bool_moves == 1]
                #print("Play:available_moves", available_moves)
                Q_learning.CheckState(state, available_moves)
                next_action = Q_learning.SelectAction(state)
                #print("Play:q_table:\n", Q_learning.q_table, "\n")
                #print("Play:next_action: ", next_action)
                #reward = self.game.getGameEnded(board, curPlayer)
                board, curPlayer = self.game.getNextState(board, curPlayer, next_action)
                next_state = hash(board.tobytes())
                #if(curPlayer == 1):
                #the winner always will be -curPlayer(the player of the last step)
                #走到结束这一步，赢的永远是-curPlayer,不存在输的情况，输的情况仅在对方赢的state里体现
                #因为永远都是赢的下一步去判断游戏是否结束以及哪个玩家赢了
                if(self.game.getGameEnded(board, 1) == 1):
                    reward = 100
                elif(self.game.getGameEnded(board, -1) == 1):
                    reward = 100
                elif(self.game.getGameEnded(board, curPlayer) == 0 or self.game.getGameEnded(board, curPlayer) == 1e-4):
                    reward = -1
                #reward = self.game.getGameEnded(board, curPlayer)
                #else:
                    #reward = -self.game.getGameEnded(board, curPlayer)
                #print("Play:self.game.getGameEnded(board, curPlayer): ", self.game.getGameEnded(board, -curPlayer))
                #print("Play:reward", reward)
                #print("\nPlay:board\n", board, "\n")
                bool_moves = self.game.getValidMoves(board, curPlayer)
                available_moves = np.array(range(self.game.getActionSize()))[bool_moves == 1]
                Q_learning.CheckState(next_state, available_moves)
                Q_learning.UpdateQtable(state, next_state, next_action, reward, bool(reward))
        state = hash(self.board.tobytes())
        #print("The q table of each step: \n", Q_learning.q_table, "\n")
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
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

    def SelectAction(self, state):
        all_actions = self.getAvailableActions(state)
        #print("\nSelectAction: valid_actions", all_actions)
        #print("\nSelectAction: state", state)
        if(np.random.uniform() < self.epsilon):
            state_values = self.q_table.loc[state, ]
            #print("SelectAction:state_values", np.array(state_values))
            #print("SelectAction:max(state_values)", np.nanmax(state_values))
            #print("SelectAction:index", state_values==max(state_values))
            next_action = np.random.choice(state_values[state_values==np.nanmax(state_values)].index)
            #print("SelectAction:next_action:", next_action)
            #print("SelectAction:Exploitation\n")
        else:
            next_action = np.random.choice(all_actions)
            #print("SelectAction:Exploration\n")
        #print("SelectAction:next_action ", next_action)
        return(next_action)

    def getAvailableActions(self, state):
        available_actions = []
        #print("getAvailableActions: self.q_table.index", np.array(self.q_table.columns))
        for n,i in enumerate(self.q_table.loc[state,]):
            if(math.isnan(i) == False):
                #print("getAvailableActions:unvalid moves: ", i)
                available_actions.append(self.q_table.columns[n])
        #print("getAvailableActions: q_table\n", self.q_table)
        return available_actions

    def UpdateQtable(self, state, next_state, action, reward, game_end):
        q = self.q_table.loc[state, action]
        if(game_end == False):
            q_ = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_ = reward
        #print("UpdateQtable:q_", q_)
        #print("UpdateQtable:reward", reward)
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
            #print("\nCheckState:q_table", self.q_table.columns)
            #print("\nCheckState:available_actions", available_actions)
            #print("\nCheckState:test", np.setdiff1d(self.q_table.columns, available_actions))
            unvalid_moves = np.setdiff1d(self.q_table.columns, available_actions)
            for m in unvalid_moves:
                self.q_table.loc[state, m] = None
