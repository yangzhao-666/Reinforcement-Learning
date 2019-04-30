'''
write MCTS player by yourself
'''
import numpy as np

class MCTS():
    '''
    nun_of_steps: The number of simulations for each node
    '''
    def __init__(self, game, num_of_steps = 20):
        self.game = game
        self.nos = num_of_steps

    def play(self, board):
        tree = Tree(board, self.game)
        return tree.Start(self.nos)


class TreeNode():
    '''
    Q: Total simulation reward (number of winning);
    N: Total number of visits;
    state: If it's already been visited, 0 is no, 1 is yes;
    getNextNode(): If the node is not fully expanded, select next unvisited node;
    SelectNextNode(): Get the node of next step, includes 2 kinds of ndoes, one if from UCT, one is just pick;
    '''
    def __init__(self, board, game):
        self.board = board
        self.game = game
        self.Q = 1
        self.N = 1
        self.children = []
        self.state = 0
        self.parent = []
        self.curPlayer = 1
        self.action = []

    def isFullyExpanded(self):
        for node in self.children:
            if(node.state == 0):
                return False
        return True

    def Expand(self):
        bool_valids_moves = self.game.getValidMoves(self.board, self.curPlayer)
        valids_moves = np.array(range(self.game.getActionSize()))[bool_valids_moves == 1]
        valids_moves_size = len(valids_moves)
        for i, move in enumerate(valids_moves):
            tmp_board = self.board
            #print("Expand: move: ", move)
            tmp_board, _ = self.game.getNextState(tmp_board, self.curPlayer, int(move))
            child = TreeNode(tmp_board, self.game)
            child.parent = self
            child.curPlayer = -self.curPlayer
            #print("Expand:self.player: ", self.curPlayer, " child.curPlayer: ", child.curPlayer)
            child.action = move
            self.children.append(child)

    def SelectNextNode(self):
        if(self.game.getGameEnded(self.board, self.curPlayer) == 0 and self.children == []):
            self.Expand()
        if(self.isFullyExpanded()):
            return self.BestUCT()
        elif(self.game.getGameEnded(self.board, self.curPlayer) != 0):
            return self
        else:
            return self.getNextChild()

    def BestUCT(self):
        uct = [0] * len(self.children)
        c = np.sqrt(2.0)
        for i, child in enumerate(self.children):
            uct[i] = child.Q / child.N + c * np.sqrt(2 * np.log(self.N) / child.N)
        next_node = self.children[uct.index(max(uct))]
        if(next_node.game.getGameEnded(next_node.board, next_node.curPlayer) != 0):
            return next_node
        else:
            return next_node.SelectNextNode()

    def getNextChild(self):
        for child in self.children:
            if(child.state == 0):
                child.state = 1
                return child

    def BP(self, winner):
        self.N = self.N + 1
        if(self.curPlayer == winner):
            self.Q = self.Q + 0
            #print("BP: self")
        else:
            self.Q = self.Q + 1
            #print("BP: opposite")
        if(self.parent != []):
            self.parent.BP(winner)

    # According to the policy, given a current board and return next move;
    # This is a policy function.
    def RollOut(self, num):
        for i in range(num):
            tmp_player = self.curPlayer
            tmp_board = self.board
            while(self.game.getGameEnded(tmp_board, tmp_player) == 0):
                a = np.random.randint(self.game.getActionSize())
                valids = self.game.getValidMoves(tmp_board, tmp_player)
                while valids[a] != 1:
                    a = np.random.randint(self.game.getActionSize())
                #print("RollOut:a ", a)
                #print("RollOut:a", type(a))
                tmp_board, tmp_player = self.game.getNextState(tmp_board, tmp_player, int(a))
                #print("RollOut:tmp_player ", tmp_player)
                #print("RollOut:tmp_board:\n", tmp_board)
            if(tmp_player == -1):
                winner = 1
            else:
                winner = -1
        #print("RollOut: winner", winner)
        return winner

    def BestAction(self):
        node = TreeNode(self.board, self.game)
        for child in self.children:
            if(child.N >= node.N):
                node = child
        return node, node.action

    def Start(self):
        node = self.SelectNextNode()
        if(node.game.getGameEnded(node.board, node.curPlayer) == 0):
            winner = node.RollOut(1)
            node.BP(winner)
        else:
            if(node.game.getGameEnded(node.board, node.curPlayer) == 1):
                node.BP(node.curPlayer)
            elif(node.game.getGameEnded(node.board, node.curPlayer) == -1):
                node.BP(-node.curPlayer)

    def traverse(self):
        for child in self.children:
            print("Q: ", child.Q, "    N: ", child.N, "\n", child.board, "\n current player: ", child.curPlayer)
            child.traverse()

class Tree():
    def __init__(self, board, game):
        self.root = TreeNode(board, game)
        self.game = game
        self.root.curPlayer = 1

    def traverse(self):
        return self.root.traverse()

    def Start(self, num):
        while(num > 0):
            self.root.Start()
            num = num - 1
        node, ret = self.root.BestAction()
        #self.traverse()
        #print("\nNode\n", node)
        #print("Next Action: ", ret, "\nNode\n", node, node.parent)
        return ret
