import Arena
#from connect4.Connect4Game import Connect4Game, display
#from connect4.Connect4Players import *
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
#from gobang.GobangGame import GobangGame, display
#from gobang.GobangPlayers import *
import numpy as np
from MCS import *
from MCTS import *
from Qlearning import *
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
#define games
#g = Connect4Game(4)
g = TicTacToeGame()
#g = GobangGame(7)

#define players
rp = RandomPlayer(g).play
mcsp = MCS(g,100).play
mctsp = MCTS(g,100).play
qlp = Qlearning(g, 100, 0.01, 0.9, 0.9).play

arena_rp_op = Arena.Arena(mcsp, rp, g, display=display)
print(arena_rp_op.playGames(10, verbose=False))
