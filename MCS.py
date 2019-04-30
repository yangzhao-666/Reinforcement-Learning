'''
write MCS player as a small exercise before you implement the MCTS
'''
import numpy as np

class MCS():
    def __init__(self, game, iterations = 100):
        self.game = game
        self.iterations = iterations

    # This functino retruns an action
    def play(self, board):
        bool_valids_moves = self.game.getValidMoves(board, 1)
        valids_moves = np.array(range(self.game.getActionSize()))[bool_valids_moves == 1]
        num_wins = [0] * self.game.getActionSize()
        og_board = board
        for i, move in enumerate(valids_moves):
            curPlayer = 1
            board, _ = self.game.getNextState(board, curPlayer, move)
            iterations = self.iterations
            while(iterations > 0):
                tmp_board = board
                winner = self.SimulateOnce(board)
                iterations = iterations - 1
                board = tmp_board
                if(winner == 1):
                    num_wins[i] += 1
            board = og_board
        return(valids_moves[num_wins.index(max(num_wins))])

    # Return the winner, 1 is MCS player, -1 is opposite player
    def SimulateOnce(self, board):
        curPlayer = -1
        while(self.game.getGameEnded(board, curPlayer) == 0):
            a = np.random.randint(self.game.getActionSize())
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            while valids[a] != 1:
                a = np.random.randint(self.game.getActionSize())
            board, curPlayer = self.game.getNextState(board, curPlayer, a)
        if(curPlayer == 1):
            return -1
        elif(curPlayer == -1):
            return 1
