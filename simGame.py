import numpy as np
import pickle
import random
import math
import sys
import matplotlib.pyplot as plt
from itertools import combinations, product
from ast import literal_eval
import csv



def getBoards(d):
# =============================================================================
# 	Returns a list of boards with d blanks (distance d from a full board)
#This is useful because we must calculate Richman and discrete-Richman
# 	values backwards-recursively (that is, the Richman value of any state
# 	is determined by the Richman values of its child states). With the
#   states partitioned by distance from a full state, and with the knowledge
#   that any child states of a given gamestate must be 1 closer to a full
#   state, we can systematically calculate the Richman values for all
#   terminal nodes (that is, all states in distance0), then for all states
#   in distance1, then in distance2, and so on.
# =============================================================================
    if d == 15:
        return [Board()]
    
    boards = []
    upper = 15-d
    fillup = (15-d)//2+1
    allPos = product(range(6), repeat=2)
    allPos = [(a,b) for (a,b) in allPos if not a<=b]
    for i in range(fillup):
        p1Pos = list(combinations(allPos, i))
        #print(list(p1Pos))
        p2Pos = list(combinations(allPos, upper-i))
        #print(p2Pos)
        for edges1 in p1Pos:
            for edges2 in p2Pos:
                if (set(edges1)).isdisjoint(set(edges2)):
                    boards.append(Board([list(edges1), list(edges2)]))
                    boards.append(Board([list(edges2), list(edges1)]))
    return boards



class ActionI:
    pass
    
class AgentI:
    def chooseAction(self, positions, state):
        pass
    
    def endGame(self, state):
        pass
    
    def reset(self):
        pass
    
    def feedReward(self, reward):
        pass
    
    def addState(self, state):
        pass
    
    
class GameI:
    def play(self, state):
        pass
    

class StateI:
    def getHash(self):
        pass
    
    def updateState(self, action):
        pass
    
    def giveReward(self):
        pass
    
    def reset(self):
        pass
    
    
class BoardI:
    def getHash(self):
        pass
    
    def winner(self):
        pass
    
    def availableActions(self):
        pass


class Board(BoardI):
        

    def __init__(self, config=None): # store positions with 1 and positions with 2
        if config is None:
            self.board = [[],[]]
        else:
            self.board = config
        
    def fromArray(self, arr):
        self.board = arr
        
    # get unique hash of current board state
    def getHash(self):
        self.board[0].sort()
        self.board[1].sort()
        return hash(str(self.board))
    
    def winner(self):
        p1Edges = self.board[0]
        p2Edges = self.board[1]
        
        for v1, v2 in p1Edges:
            for v3 in range(6):
                if not v3==v1 and not v3==v2:
                    if ((v1,v3) in p1Edges or (v3,v1) in p1Edges) and ((v2,v3) in p1Edges or (v3,v2) in p1Edges):
                        return -1
        for v1, v2 in p2Edges:
            for v3 in range(6):
                if not v3==v1 and not v3==v2:
                    if ((v1,v3) in p2Edges or (v3,v1) in p2Edges) and ((v2,v3) in p2Edges or (v3,v2) in p2Edges):
                        return 1
#        tuples = list(combinations(range(6), 3))
#        for v1, v2, v3 in tuples:
#            if self.board[v1, v2] > 0:
#                potentialLoser = self.board[v1, v2]
#                if self.board[v1, v3] == potentialLoser and self.board[v2, v3] == potentialLoser:
#                    self.isEnd = True
#                    return -1 * potentialLoser
        # not end
        if len(p1Edges)+len(p2Edges)==15:
            return 0
        return None
    
    def numBlanks(self):
        return (15-len(self.board[0])-len(self.board[1]))
    
    def availableActions(self):
        positions = []
        allPos = product(range(6), repeat=2)
        allPos = [(a,b) for (a,b) in allPos if not a<=b]
        for v1, v2 in allPos:
            if not (v1,v2) in self.board[0] and not (v1,v2) in self.board[1]:
                    positions.append((v1, v2))  # need to be tuple
        return positions
    
    def getBoard(self):
        return [list.copy(self.board[0]), list.copy(self.board[1])]
    
    def copy(self):
        copy = Board()
        copy.fromArray(self.getBoard())
        return copy
    
    def nextBoards(self, symbol):
        if (self.winner() is not None):
            return []
        children = []
        if symbol==1:
            token = 0
        else:
            token = 1
        for v1, v2 in self.availableActions():
            child = self.getBoard()
            child[token].append((v1,v2))
            b = Board()
            b.fromArray(child)
            children.append(b) 
        return children
    
    def generateMove(self, nextBoard):
        nextB = nextBoard.getBoard()
#        tuples = list(combinations(range(6), 2))
        diff1 = [pos for pos in nextB[0] if pos not in self.board[0]]
        diff2 = [pos for pos in nextB[1] if pos not in self.board[1]]
        if len(diff1) == 0:
            return diff2[0]
        else:
            return diff1[0]
#        for v1, v2 in tuples:
#            if self.board[v1, v2] != nextB[v1, v2]: 
#                return v1, v2
#        return -1,-1

    def updateState(self, position, symbol):
        if symbol==1:
            self.board[0].append(position)
            #self.board[0].sort()
        else:
            self.board[1].append(position)
            #self.board[1].sort()
        
    def showBoard(self):
        print(self.board)
    
    def standardString(self, symbol):
        if symbol==1:
            return str(self.getBoard())
        else:
            return str([self.board[1], self.board[0]])
    
    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.board, other.board)



class State(StateI):

    def __init__(self, p1, p2, totalChips):
        self.board = Board()
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        #agentChips = math.ceil(0.51953126*totalChips)
        agentChips = math.ceil(0.5*totalChips)
        self.chips = [totalChips, agentChips, totalChips-agentChips]
        self.tieBreaker = 1
        
    def copy(self):
        p1 = self.p1
        p2 = self.p2
        copy = State(p1, p2, self.chips[0])
        copy.chips = [self.chips[0], self.chips[1], self.chips[2]]
        copy.playerSymbol = self.playerSymbol
        copy.isEnd = self.isEnd
        copy.board = self.board.copy()
        copy.boardHash = self.boardHash
        copy.tieBreaker = self.tieBreaker
        return copy

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = self.board.getHash()
        return str(self.boardHash) + str(self.tieBreaker) + str(self.playerSymbol) #+ str(self.chips[1]) + str(self.chips[2]) 

    # only when game ends
    def giveReward(self):
        result = self.board.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = Board()
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        #agentChips = math.ceil(0.51953126*self.chips[0])
        agentChips = math.ceil(0.5*self.chips[0])
        self.chips = [self.chips[0], agentChips, self.chips[0]-agentChips]
        
    def bid(self): 
        p1Bid = self.p1.getBid(self, 1, 2, self.chips[1])
        p2Bid = self.p2.getBid(self, 2, 1, self.chips[2])
        #print(p1Bid)
        #print(p2Bid)
        useTb = 0
        if p1Bid % 1 == 0.25:
            if self.tieBreaker == 1:
                p1Bid = math.floor(p1Bid)
                self.p1.addBid(p1Bid)
                useTb = 1
            else:
                if p1Bid < self.chips[1]:
                    p1Bid = math.ceil(p1Bid)
                    self.p1.addBid(p1Bid)
        if p2Bid % 1 == 0.25:
            if self.tieBreaker == -1:
                p2Bid = math.floor(p2Bid)
                self.p2.addBid(p2Bid)
                useTb = 1
            else:
                if p2Bid < self.chips[2]:
                    p2Bid = math.ceil(p2Bid)
                    self.p2.addBid(p2Bid)
        p1Bid = min(p1Bid, self.chips[1])
        p2Bid = min(p2Bid, self.chips[2])
        self.chips[1] = self.chips[1] - p1Bid  + p2Bid
        self.chips[2] = self.chips[2] - p2Bid + p1Bid
        if (p1Bid > p2Bid):
            self.playerSymbol = 1
            self.p1.addStateBidProb(self, p1Bid, 1, 0)
            self.p2.addStateBidProb(self, p2Bid, 0, 0)
            return 1 
        elif (p1Bid < p2Bid):
            self.playerSymbol = -1
            self.p1.addStateBidProb(self, p1Bid, 0, 0)
            self.p2.addStateBidProb(self, p2Bid, 1, 0)
            return -1
        else:
            
            if useTb == 1:
                winner = self.tieBreaker
                if winner == 1:
                    self.p1.addStateBidProb(self, p1Bid, 1, 1)
                    self.p2.addStateBidProb(self, p2Bid, 0, 0)
                else:
                    self.p1.addStateBidProb(self, p1Bid, 0, 0)
                    self.p2.addStateBidProb(self, p2Bid, 1, 1)
                self.tieBreaker *= -1
            else:
                winner = -1 * self.tieBreaker
                if winner == 1:
                    self.p1.addStateBidProb(self, p1Bid, 1, 0)
                    self.p2.addStateBidProb(self, p2Bid, 0, 0)
                else:
                    self.p1.addStateBidProb(self, p1Bid, 0, 0)
                    self.p2.addStateBidProb(self, p2Bid, 1, 0)
            self.playerSymbol = winner
            return winner
# =============================================================================
#             if np.random.uniform(0, 1) <= 0.5:
#                 self.playerSymbol = 1
#                 return 1 
#             else:
#                 self.playerSymbol = -1
#                 return -1
# =============================================================================
    def win(self):
        w = self.board.winner()
        if not w==None:
            self.isEnd = True
        return w
    
    def updateState(self, position):
        self.board.updateState(position, self.playerSymbol)
        # switch to another player
        # self.playerSymbol = -1 if self.playerSymbol == 1 else 1
            


class Player(AgentI):
    def __init__(self, name, prob, biddingStrategy, symbol, totalChips, learn_rate=0.4, exp_rate=0.7):
        self.name = name
        self.states = []  # record all positions taken
        self.bids = []  # record all bids taken
        self.lr = learn_rate
        self.exp_rate = exp_rate
        self.exp_rate_decay = 0.8
        self.decay_gamma = 0.8
        self.states_value = {}  # state -> value
        self.biddingStrategy = biddingStrategy
        self.symbol = symbol
        self.next_action = None
        self.data = {}
        self.totalChips = totalChips
        self.prob = prob
        self.stateProbs = {} # estimate prob of winning bid in state s with bid b - dict from state hash and bid to (sum of bids won, number of bids)

        # nodesToDiscreteRich is a list of dictionaries that hold
		# entries of the form
		# gameNode : discrete-Richman value.
		#
		# The nodes are partitioned by their distance away from a
		# full state, such that nodesToDiscreteRich[k] carries all
		# entries with node-keys that are k steps away from a full
		# state.
        self.nodesToDiscreteRich = [{} for i in range(16)]

		# nodesToMoveBid is a list of dictionaries that hold
		# entries of the form
		# gameNode : (optimalMove, optimalBid),
		#
		# where optimalMove is of the form (row, col). The nodes 
		# are partitioned by their distance away from a full state
        self.nodesToMoveBid = [{} for i in range(16)]

        if biddingStrategy == "optimal" or biddingStrategy == "optimalExp":
            fd = open('simDiscreteRich', 'rb')
            self.nodesToDiscreteRich = pickle.load(fd)
            fd.close()
            fm = open('simMoveBid', 'rb')
            self.nodesToMoveBid = pickle.load(fm)
            fm.close()
            


    def getHash(self, board):
        boardHash = str(board.reshape(3,3)) # + str(board.chipsP1) + str(board.chipsP2)
        return boardHash
    
    def randomBid(self, availableTokens, tieBreaker, symbol):
        tb = 0
        if (tieBreaker==symbol):  #have tiebreaker
            if np.random.uniform(0, 1) <= 0.5:
                tb = 1
        return (random.randint(0, availableTokens) + 0.25*tb, tb)
    
    def randomBidAndAction(self, availableTokens, tieBreaker, symbol, positions):
        tb = 0
        if (tieBreaker==symbol):  #have tiebreaker
            if np.random.uniform(0, 1) <= 0.5:
                tb = 1
        idx = np.random.choice(len(positions))
        action = positions[idx]
        return (random.randint(0, availableTokens) + 0.25*tb, tb, action)
    
    def getProb(self, stateHash, b, tb):
        if self.prob == 'simple':
            (bidsWon, bidsNum) = self.stateProbs.get((stateHash, b, tb), (0,1))
            return (bidsWon) / (bidsNum)
        elif self.prob == 'laplace':
            (bidsWon, bidsNum) = self.stateProbs.get((stateHash, b, tb), (0,0))
            return (bidsWon + 1) / (bidsNum + 2)
    
    def addStateBidProb(self, state, bid, win, tb):
        (won, num) = self.stateProbs.get((state, bid, tb), (0,0))
        self.stateProbs[state, bid, tb] = (won+win, num+1)
    
    def stateValBid(self, prob, action, value_max, bid, tb, pos, pId, oId, availableTokens, tieBreaker, symbol, state):
        for b in range(int(availableTokens) + 1):
            if (tieBreaker==symbol):  #have tiebreaker
                next_state = state.copy()
                next_state.tieBreaker *= -1
                next_state.playerSymbol = symbol
                if not pos is None:
                    next_state.board.updateState(pos, symbol)
                next_stateHash = next_state.getHash()
                value = self.states_value.get((next_stateHash), 0)
                if len(prob) > 0:
                    bidWinProb = self.getProb(state.getHash(), b, 1)
                    value *= bidWinProb
                if value > value_max:
                    value_max = value
                    bid = b
                    tb = 1
                    action = pos
            next_state = state.copy()
            next_state.chips[pId] = max(next_state.chips[pId] - 1, 0)
            next_state.chips[oId] = min(next_state.chips[oId] + 1, next_state.chips[0])
            next_state.playerSymbol = symbol
            if not pos is None:
                next_state.board.updateState(pos, symbol)
            next_stateHash = next_state.getHash()
            value = self.states_value.get((next_stateHash), 0)
            if len(prob) > 0:
                bidWinProb = self.getProb(state.getHash(), b, 0)
                value *= bidWinProb
            if value > value_max:
                value_max = value
                bid = b
                tb = 0
                action = pos
        return (value_max, action, bid, tb)
    
    def actionValBid(self, prob, action, value_max, bid, tb, pos, pId, oId, availableTokens, tieBreaker, symbol, state):
        for b in range(int(availableTokens) + 1):
            if (tieBreaker==symbol):  #have tiebreaker
                next_state = state.copy()
                next_state.tieBreaker *= -1
                next_state.playerSymbol = symbol
                if not pos is None:
                    next_state.board.updateState(pos, symbol)
                next_stateHash = next_state.getHash()
                value = self.states_value.get((next_stateHash, b), 0)
                if len(prob) > 0:
                    bidWinProb = self.getProb(state.getHash(), b, 1)
                    value *= bidWinProb
                if value > value_max:
                    value_max = value
                    bid = b
                    tb = 1
                    action = pos
            next_state = state.copy()
            next_state.chips[pId] = max(next_state.chips[pId] - 1, 0)
            next_state.chips[oId] = min(next_state.chips[oId] + 1, next_state.chips[0])
            next_state.playerSymbol = symbol
            if not pos is None:
                next_state.board.updateState(pos, symbol)
            next_stateHash = next_state.getHash()
            value = self.states_value.get((next_stateHash, b), 0)
            if len(prob) > 0:
                    bidWinProb = self.getProb(state.getHash(), b, 0)
                    value *= bidWinProb
            if value > value_max:
                value_max = value
                bid = b
                tb = 0
                action = pos
        return (value_max, action, bid, tb)
    
    def getBid(self, state, pId, oId, availableTokens):
        self.next_action = None
        positions = state.board.availableActions()
        prob = self.prob
        if (self.biddingStrategy == "random"):
            # do random bid
            (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
            self.next_action = action
            self.data[(state.board.standardString(self.symbol), tb)] = bid
            return bid
        elif (self.biddingStrategy == "state-value1" or self.biddingStrategy == "TD" or self.biddingStrategy == "pureTD"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                self.next_action = action
                return bid
            else:
                # do greedy bid
                value_max = -999
                bid = 0
                tb = 0
                action = None
                for pos in positions:
                    (value_max, action, bid, tb) = self.stateValBid(prob, action, value_max, bid, tb, pos, pId, oId, availableTokens, state.tieBreaker, self.symbol, state)
                self.next_action = action
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "state-value2"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                self.next_action = action
                return bid
            else:
                # do greedy bid
                value_max = -999
                bid = 0
                tb = 0
                action = None
                (value_max, action, bid, tb) = self.stateValBid(prob, action, value_max, bid, tb, None, pId, oId, availableTokens, state.tieBreaker, self.symbol, state)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "action-value1"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                self.next_action = action
                return bid
            else:
                # do greedy bid
                value_max = -999
                bid = 0
                tb = 0
                action = None
                for pos in positions:
                    (value_max, action, bid, tb) = self.actionValBid(prob, action, value_max, bid, tb, pos, pId, oId, availableTokens, state.tieBreaker, self.symbol, state)
                self.next_action = action
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "action-value2"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                self.next_action = action
                return bid
            else:
                # do greedy bid
                value_max = -999
                bid = 0
                tb = 0
                stateHash = state.getHash()
                # no afterstates
                for b in range(int(availableTokens) + 1):
                    if (state.tieBreaker==self.symbol):  #have tiebreaker
                        value = self.states_value.get((stateHash), 0)
                        if len(prob) > 0:
                            bidWinProb = self.getProb(state.getHash(), b, 1)
                            value *= bidWinProb
                        if value >= value_max:
                            value_max = value
                            bid = b
                            tb = 1
                    value = self.states_value.get((stateHash), 0)
                    if len(prob) > 0:
                        bidWinProb = self.getProb(state.getHash(), b, 0)
                        value *= bidWinProb
                    if value >= value_max:
                        value_max = value
                        bid = b
                        tb = 0
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "optimal"):
            numBlanks = state.board.numBlanks()
            move, bid = self.nodesToMoveBid[numBlanks][state.board.getHash()]
            self.next_action = move
            tb = 0
            if state.tieBreaker==self.symbol and bid % 1 == 0.25:
                tb = 1
            self.data[(state.board.standardString(self.symbol), tb)] = bid
            return bid + 0.25
        elif (self.biddingStrategy == "optimalExp"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                (bid, tb, action) = self.randomBidAndAction(availableTokens, state.tieBreaker, self.symbol, positions)
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                self.next_action = action
                return bid
            else:
                # do greedy bid
                numBlanks = state.board.numBlanks()
                move, bid = self.nodesToMoveBid[numBlanks][state.board.getHash()]
                tb = 0
                self.next_action = move
                if state.tieBreaker==self.symbol and bid % 1 == 0.25:
                    tb = 1
                self.data[(state.board.standardString(self.symbol), tb)] = bid
                return bid
                    

    def generateStrategy(self, totalChips):

# =============================================================================
# 		This method populates nodesToDiscreteRich and nodesToMoveBid.
# =============================================================================

        print("\tGenerating strategy...")

# =============================================================================
# 		BASE CASES:
# 		
# 		We first assign discrete-Richman values to all terminal nodes.
# 		All terminal nodes are guaranteed to be a win (for the agent),
# 		a draw, or a loss.  The theory defines the disrete-Richman value
# 		of any win state to be 0 (that is, you need 0 chips to win from
# 		that state), and the discrete-Richman value for any draw or loss
# 		to be k+1, where k is the total number of chips in play. 
# 		See the paper titled 'Discrete Bidding Games' by Develin and Payne 
# 		for more details.
# =============================================================================

        boards0 = getBoards(0)
        for board in boards0:
            if board.winner()==self.symbol:
                self.nodesToDiscreteRich[0][board.getHash()] = 0.0
            else:
                self.nodesToDiscreteRich[0][board.getHash()] = totalChips + 1.0        


# =============================================================================
# 		BACKWARDS INDUCTION:
# 		Calculating discrete-Richman values is similar to calculating Richman
# 		values, but needs to check different cases to maintain discreteness.  As 
# 		these cases are technical and would take a good amount of space to
# 		explain, we again refer the reader to the paper 'Discrete Bidding Games'
# 		by Develin and Payne for a better explanation.  Having read that, the
# 		relatively brief documentation in the four cases toward the end of the loop
# 		below should suffice.
# =============================================================================
				
        for i in range(1,16):
			# Get all nodes that are i steps away from a full state
            print(i)
            nodes = getBoards(i)

            for node in nodes:
                
				# If the node is a win state for the agent, assign it 
				# a discrete-Richman value of 0.
                winner = node.winner()
                if winner==self.symbol:
                    self.nodesToDiscreteRich[i][node.getHash()] = 0.0
                    continue
				# Else if the node is a win state for the opponent, assign
				# a discrete-Richman value of k+1.
                elif winner==(-1*self.symbol):
                    self.nodesToDiscreteRich[i][node.getHash()] = totalChips + 1.0
                    continue

				# For the current node, find the minimum discrete-Richman
				# value of its children that the agent can move to, and the
				# maximum discrete-Richman value of its children that the
				# opponent can move to.  As well, store the child node that
				# corresponds to the minimum discrete-Richman value, so that
				# we can determine the optimal move.
				
                Fmax = -1.0
                Fmin = sys.maxsize
                myChildren = node.nextBoards(self.symbol)
                oppChildren = node.nextBoards(-1*self.symbol) 
                        
                for myChild in myChildren:
                    if Fmin > self.nodesToDiscreteRich[i-1][myChild.getHash()]:
                        Fmin = self.nodesToDiscreteRich[i-1][myChild.getHash()]
                        favoredChild = myChild

                for oppChild in oppChildren:
                    Fmax = max(Fmax,self.nodesToDiscreteRich[i-1][oppChild.getHash()])

				# Discrete-Richman values may or may not include the tie-breaking
				# chip *.  Conveniently, as the value of * is strictly positive but
				# strictly less than 1 (i.e. 0 < * < 1), we can encode these into
				# the discrete-Richman values as a decimal part of 0.5.  This may
				# be included in the final calculated value of the discrete-Richman
				# value of the current node depending on the cases mentioned above,
				# and can be seen in the value of epsilon.  FmaxVal and FminVal
				# store the underlying integer value of Fmax and Fmin, respectively.
				# Note that Payne and Develin denote underlying value of a discrete-
				# Richman value with absolute value bars.

				# Note that, in two of the cases, the optimal bid is appended with
				# a decimal part of 0.25.  This should not be thought of as part of
				# the value of the optimal bid, but rather as marker.  If the agent
				# makes a bid of the form n + 0.25, this means that the game engine
				# should check if the agent has the tie breaking chip at that moment.
				# If the agent does, then the agent will bet n (and use the tie
				# breaking chip if a tie arises); if the agent does not have the
				# tie breaking chip, then the agent will bet n+1.  
			    	
                FmaxVal = math.floor(Fmax)
                FminVal = math.floor(Fmin)
                Fsum = FmaxVal + FminVal
				# If Fsum is odd and Fmin \in \N*
                if (Fsum % 2 == 1) and FminVal < Fmin:
                    epsilon = 1.0
                    bid = math.floor(abs(FmaxVal-FminVal)/2.0) * 1.0
				# Else if Fsum is odd and Fmin \in \N
                elif (Fsum % 2 == 1) and FminVal == Fmin:
                    epsilon = 0.5
                    bid = math.floor(abs(FmaxVal-FminVal)/2.0) + 0.25

				# Else if Fsum is even and Fmin \in \N*
                elif (Fsum % 2 == 0) and FminVal < Fmin:
                    epsilon = 0.5
                    #bid = 0
                    bid = max(0,abs(FmaxVal-FminVal)/2.0 - 0.75)

				# Else (i.e., if Fsum is even and Fmin \in \N)
                else:
                    epsilon = 0.0
                    bid = abs(FmaxVal-FminVal)/2.0
                
                self.nodesToDiscreteRich[i][node.getHash()] = math.floor(Fsum/2.0) + epsilon
                self.nodesToMoveBid[i][node.getHash()] = (node.generateMove(favoredChild), bid)         
                        
    

    def chooseAction(self, positions, current_state):
        if (self.next_action == None):
            value_max = -999
            for p in positions:
                next_state = current_state.copy()
                #next_state.playerSymbol = self.symbol
                next_state.board.updateState(p, self.symbol)
                next_stateHash = next_state.getHash()
                value = self.states_value.get(next_stateHash, 0)
                if value > value_max:
                    value_max = value
                    action = p
            return action
        else:
            return self.next_action
        
        
    # append a hash state
    def addState(self, state):
        self.states.append(state)
        
       # append a bid
    def addBid(self, bid):
        self.bids.append(bid)
        
    # after each round, backpropagate and update state values
    def update(self, state, next_state, me):
        if self.biddingStrategy == "TD" or self.biddingStrategy == "pureTD":
            st_hash = state.getHash()
            next_st_hash = next_state.getHash()
            if self.states_value.get(st_hash) is None:
                    if state.board.winner() == me:
                        self.states_value[st_hash] = 1
                    elif state.board.winner() == -1 * me:
                        self.states_value[st_hash] = 0
                    else:
                        self.states_value[st_hash] = 0.5
            if self.states_value.get(next_st_hash) is None:
                    if next_state.board.winner() == me:
                        self.states_value[next_st_hash] = 1
                    elif next_state.board.winner() == -1 * me:
                        self.states_value[next_st_hash] = 0
                    else:
                        self.states_value[next_st_hash] = 0.5
            self.states_value[st_hash] += self.lr * (self.decay_gamma * self.states_value[next_st_hash] - self.states_value[st_hash])
        

    # at the end of game, backpropagate and update state value
    def feedReward(self, reward):
        if self.biddingStrategy == "action-value1" or self.biddingStrategy == "action-value2":
            for st, b in zip(reversed(self.states), reversed(self.bids)):
                if self.states_value.get((st, b)) is None:
                    self.states_value[(st, b)] = 0.5
                self.states_value[(st, b)] += self.lr * (self.decay_gamma * reward - self.states_value[(st, b)])
                reward = self.states_value[st, b]
                #elif not self.biddingStrategy == "TD":
        elif not self.biddingStrategy == "pureTD":
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0.5
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]

    def reset(self):
        self.states = []
        self.nextAction = None

    def savePolicy(self):
        fw = open('sim_policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer(AgentI):
    def __init__(self, name):
        self.name = name
        
    def getBid(self, state, pId, availableTokens):
        while True:
            bid = int(input("You have " + str(availableTokens) + " tokens. Input your bid: "))
            if bid <= availableTokens:
                if state.tieBreaker == -1:
                    tieBreak = int(input("Use tiebreaker in case of tie? (1 for yes, 0 for N) "))
                    return (bid + tieBreak*0.25)
                else:
                    return (bid)
            else:
                print("You do not have enough tokens.")

    def chooseAction(self, positions, current_state):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass
    
class Sim(GameI):
    
    def __init__(self):
        self.p1Dicts = []
        self.p2Dicts = []
        self.p1Win = 0
        self.p2Win = 0
        self.wins = []
    
    def play(self, init_state, rounds=100):
        state = init_state
        for i in range(rounds):
            if i > 0 and i % 100 == 0:
                self.p1Dicts.append(state.p1.data.copy())
                self.p2Dicts.append(state.p2.data.copy())
                self.wins.append((self.p1Win)/i)
            if i % 1000 == 0:
                state.p1.exp_rate *= state.p1.exp_rate_decay
                state.p2.exp_rate *= state.p2.exp_rate_decay
                print("Rounds {}".format(i))
                #print(sum(state.p1.states_value.values()))

            while not state.isEnd:
                #state.board.showBoard()
                old_state = state.copy()
                turn = state.bid()
                if turn == 1:
                    # Player 1
                    state_hash = state.getHash()
                    state.p1.addState(state_hash)
                    state.p2.addState(state_hash)
                    
                    positions = state.board.availableActions()
                    p1_action = state.p1.chooseAction(positions, state)
                    # take action and upate board state
                    state.updateState(p1_action)
                    
                    state.p1.update(old_state, state, state.p1.symbol)
                    state.p2.update(old_state, state, state.p2.symbol)
                    
                    # check board status if it is end

                    win = state.win()
                    if win is not None:
                        #state.board.showBoard()
                        # ended with p2 either win or draw
                        self.p2Win += 1
                        state.giveReward()
                        state.p1.reset()
                        state.p2.reset()
                        state.reset()
                        break
                    
                else:
                    # Player 2
                    state_hash = state.getHash()
                    state.p1.addState(state_hash)
                    state.p2.addState(state_hash)
                    
                    positions = state.board.availableActions()
                    p2_action = state.p2.chooseAction(positions, state)
                    state.updateState(p2_action)
                    
                    state.p1.update(old_state, state, state.p1.symbol)
                    state.p2.update(old_state, state, state.p2.symbol)
                    

                    win = state.win()
                    if win is not None:
                        #state.board.showBoard()
                        # ended with p2 either win or draw
                        self.p1Win += 1
                        state.giveReward()
                        state.p1.reset()
                        state.p2.reset()
                        state.reset()
                        break
        self.p1Dicts.append(state.p1.data.copy())
        self.p2Dicts.append(state.p2.data.copy())
        self.wins.append((self.p1Win)/rounds)

    # play with human
    def play2(self, state):
        while not state.isEnd:
            turn = state.bid()
            if turn == 1:
                # Player 1
                positions = state.board.availableActions()
                p1_action = state.p1.chooseAction(positions, state)
                # take action and upate board state
                state.updateState(p1_action)
                print("Human: " + str(state.chips[2]) + "   Computer: " + str(state.chips[1]))
                if (state.tieBreaker == 1):
                    print("Computer has the tiebreaker.")
                else:
                    print("Human has the tiebreaker.")
                state.board.showBoard()
                # check board status if it is end
                win = state.win()
                if win is not None:
                    if win == 1:
                        print(state.p1.name, "wins!")
                    else:
                        print("tie!")
                    state.reset()
                    break

            else:
                # Player 2
                positions = state.board.availableActions()
                p2_action = state.p2.chooseAction(positions, state)

                state.updateState(p2_action)
                print("Human: " + str(state.chips[2]) + "   Computer: " + str(state.chips[1]))
                if (state.tieBreaker == 1):
                    print("Computer has the tiebreaker.")
                else:
                    print("Human has the tiebreaker.")
                state.board.showBoard()
                win = state.win()
                if win is not None:
                    if win == -1:
                        print(state.p2.name, "wins!")      
                    else:
                        print("tie!")
                    state.reset()
                    break
        
        
def StateValue():
    # training
    biddingStrategy = "state-value1"
    chips = 8
    prob = ""

    p1 = Player("p1", prob, biddingStrategy, 1, chips)
    p2 = Player("p2", prob, biddingStrategy, -1, chips)

    st = State(p1, p2, chips)
    game = Sim()
    print("training...")
    game.play(st, 50000)

    # play with human
    p1 = Player("computer", prob, biddingStrategy, 1, chips, exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p1, p2, chips)
    game.play2(st)
    
def Optimal():
    # training
    biddingStrategy = "optimal"
    chips = 8
    prob = ""

    game = Sim()
    
    # play with human
    p1 = Player("computer", prob, biddingStrategy, 1, chips, exp_rate=0)
    p2 = HumanPlayer("human")

    st = State(p1, p2, chips)
    game.play2(st)
    
    
def Plot(chips, prob, rlStrat, opt, rounds):
    rl = Player("p1", prob, rlStrat, 1, chips)
    
    rlSt = State(rl, opt, chips)
    rlGame = Sim()
    print("training...")
    rlGame.play(rlSt, rounds)
    #PlotStrats(prob, rlStrat, rl, opt)
    return PlotError2(prob, rlStrat, rlGame, opt)

def PlotBoth(chips, prob, rlStrat, rl, opt, rounds):
    if rl is None:
        rl = Player("p1", prob, rlStrat, 1, chips)
    
    rlSt = State(rl, opt, chips)
    rlGame = Sim()
    print("training...")
    rlGame.play(rlSt, rounds)
    #PlotStrats(prob, rlStrat, rl, opt
    saveStateValstoCSV(prob, rlStrat, rl)
    rl.savePolicy()
    return (PlotError2(prob, rlStrat, rlGame, opt), PlotWin(prob, rlStrat, "optimal", rlGame))


def Plot2RLs(chips, prob, strat1, strat2, rounds):
    p1 = Player("p1", prob, strat1, 1, chips)
    p2 = Player("p2", prob, strat2, -1, chips)
    
    rlSt = State(p1, p2, chips)
    rlGame = Sim()
    print("training...")
    rlGame.play(rlSt, rounds)
    #PlotStrats(prob, rlStrat, rl, opt
    saveStateValstoCSV(prob, strat1, p1)
    return (PlotError3(prob, strat1, strat2, rlGame), PlotWin(prob, strat1, strat2, rlGame))

def saveStateValstoCSV(prob, strat, agent):
    with open(prob + "_" + strat + '_simPolicy.csv', 'w', newline='') as file:
        fieldnames = ['board_hash', 'value']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
    
        writer.writeheader()
        for key in agent.states_value.keys():
            file.write("%s,%s\n"%(key,agent.states_value[key]))
            
def saveErrorstoCSV(prob, strat, errors):
    with open(prob + "_" + strat + '_simErrors.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(errors)
        
def saveWinstoCSV(prob, strat, wins):
    with open(prob + "_" + strat + '_simWins.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(wins)
      
def Wins(chips, prob, rlStrat, opt, rounds):
    rl = Player("p1", prob, rlStrat, 1, chips)
    
    rlSt = State(rl, opt, chips)
    rlGame = Sim()
    print("training...")
    rlGame.play(rlSt, rounds)
    #PlotStrats(prob, rlStrat, rl, opt)
    return PlotWin(prob, rlStrat, rlGame, opt)

def PlotWins(chips, prob, strat1, strat2, trials, rounds):
    p2 = Player("p2", prob, strat2, -1, chips)
    
    wins = []
    for i in range(trials):
        wins.append(Wins(chips, prob, strat1, p2, rounds))
    print(sum(wins)/len(wins))
    
def PlotWin(prob, strat1, strat2, rlGame):
    wins = rlGame.wins
    saveWinstoCSV(prob, strat1, wins)
    
    plt.scatter(range(len(wins)), wins)
    # Add title and axis names
    probStr = (prob + " ") if len(prob)>0 else ""
    plt.title("Sim " + probStr + strat1 + " vs " + strat2)
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean winning rate')
    plt.show()
    return wins[-1]

def boardHeuristic(value):
     pass
     

def AverageError(chips, prob, rlStrat, trials, rounds):
    opt = Player("p2", prob, "optimal", -1, chips)
    
    errors = []
    for i in range(trials):
        errors.append(Plot(chips, prob, rlStrat, opt, rounds))
    print(sum(errors)/len(errors))
    
    
def AverageErrorAndWins(chips, prob, rlStrat, trials, rounds, agent=None):
    opt = Player("p2", prob, "optimal", -1, chips)
    #print(opt.nodesToMoveBid[1])
    errors = []
    wins = []
    for i in range(trials):
        (error, win) = PlotBoth(chips, prob, rlStrat, agent, opt, rounds)
        errors.append(error)
        wins.append(win)
    print("error: ", sum(errors)/len(errors))
    print("wins: ", sum(wins)/len(wins))
    
    
def TwoRlAgents(chips, prob, strat1, strat2, trials, rounds):
    
    errors = []
    wins = []
    for i in range(trials):
        (error, win) = Plot2RLs(chips, prob, strat1, strat2, rounds)
        errors.append(error)
        wins.append(win)
    print("error: ", sum(errors)/len(errors))
    print("wins: ", sum(wins)/len(wins))


def PlotError(prob, rlStrat, rlGame, optGame):
    rlDicts = rlGame.dicts
    optDicts = optGame.dicts
    errors = []
    for i in range(len(rlDicts)):
        rDict = rlDicts[i]
        oDict = optDicts[i]
        meanError = 0.0
        
        keys = rDict.keys() & oDict.keys()
        rDict = {key: rDict[key] for key in keys}
        oDict = {key: oDict[key] for key in keys}
        (oKeys, oBids) = map(list, zip(*sorted(oDict.items(), key=boardHeuristic)))
        (rKeys, rBids) = map(list, zip(*sorted(rDict.items(), key=boardHeuristic)))
        assert oKeys == rKeys
        for key in oKeys:
            meanError += abs(float(rDict[key]) - float(oDict[key]))
        errors.append(meanError / len(oKeys))
    
    plt.scatter(range(len(errors)), errors)
    # Add title and axis names
    probStr = (prob + " ") if len(prob)>0 else ""
    plt.title("Sim " + probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]


def PlotError2(prob, rlStrat, rlGame, opt):
    rlDicts = rlGame.p1Dicts
    errors = []
    for i in range(len(rlDicts)):
        rDict = rlDicts[i]
        meanError = 0.0
        
        keys = rDict.keys()
        rDict = {key: rDict[key] for key in keys}
        (rKeys, rBids) = map(list, zip(*rDict.items()))
        for (boardStr, rlTb) in rKeys:
            b = list(literal_eval(boardStr))
            board = Board(b)
            numBlanks = board.numBlanks()
            _, optBid = opt.nodesToMoveBid[numBlanks][board.getHash()]
            #optTb = 0
            #if state.tieBreaker==self.symbol and optBid % 1 == 0.25:
            #    optTb = 1
            meanError += abs(float(rDict[(boardStr, rlTb)]) - float(optBid))
        errors.append(meanError / len(rKeys))
    
    saveErrorstoCSV(prob, rlStrat, errors)
    plt.scatter(range(len(errors)), errors)
    # Add title and axis names
    probStr = (prob + " ") if len(prob)>0 else ""
    plt.title("Sim " + probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]


def PlotError3(prob, strat1, strat2, rlGame):
    p1Dicts = rlGame.p1Dicts
    p2Dicts = rlGame.p2Dicts
    errors = []
    for i in range(len(p1Dicts)):
        dict1 = p1Dicts[i]
        dict2 = p2Dicts[i]
        meanError = 0.0
        keys = dict1.keys() & dict2.keys()
        if len(keys)==0:
            continue
        dict1 = {key: dict1[key] for key in keys}
        dict2 = {key: dict2[key] for key in keys}
        (rKeys, rBids) = map(list, zip(*dict1.items()))
        for (boardStr, rlTb) in rKeys:
            meanError += abs(float(dict1[(boardStr, rlTb)]) - float(dict2[(boardStr, rlTb)]))
        errors.append(meanError / len(rKeys))
    
    saveErrorstoCSV(prob, strat1, errors)
    plt.scatter(range(len(errors)), errors)
    # Add title and axis names
    probStr = (prob + " ") if len(prob)>0 else ""
    plt.title("Sim " + probStr + strat1 + " vs " + strat2)
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]
        

    
def PlotStrats(prob, rlStrat, rl, opt): 
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    rlDict = rl.data
    optDict = opt.data
    keys = rlDict.keys() & optDict.keys()
    rlDict = {key: rlDict[key] for key in keys}
    optDict = {key: optDict[key] for key in keys}
    (optKeys, optBids) = map(np.array, zip(*sorted(optDict.items(), key=boardHeuristic)))
    (rlKeys, rlBids) = map(np.array, zip(*sorted(rlDict.items(), key=boardHeuristic)))
    optTbs = list(zip(*(optKeys)))[1]
    rlBids1 = [rlDict[key] for key in keys if key[1]==1]
    optBids1 = [optDict[key] for key in keys if key[1]==1]
    tbs1 = [t for t in optTbs if t=='1']
    rlBids0 = [rlDict[key] for key in keys if key[1]==0]
    optBids0 = [optDict[key] for key in keys if key[1]==0]
    tbs0 = [t for t in optTbs if t=='0']
    #(rlKeys, rlBids) = np.transpose(sorted(rlDict.items(), key=boardHeuristic))
    #(rlKeys, rlBids) = sorted(rlDict.items(), key=boardHeuristic)
    #(_, optTbs) = map(np.array,list(zip(*optDict.keys())))
    #rlBids = np.array(list(rlDict.values()))
    #optBids = np.array(list(optDict.values()))
    #(rlBids, rlTbs) = map(np.array,list(zip(*rlDict.values())))
    #optBids, optTbs) = map(np.array,list(zip(*optDict.values())))
    #ax.plot_trisurf(list(keys), rlTbs, rlBids, antialiased=True)
    #ax.plot_trisurf(list(keys), optTbs, optBids, antialiased=True)
    #ax.scatter(range(len(keys)), optTbs, optBids-rlBids)
    print(len(optTbs))
    #ax.scatter(range(len(tbs0)), tbs0, np.array(optBids0)-np.array(rlBids0), c='r')
    #ax.scatter(range(len(tbs1)), tbs1, np.array(optBids1)-np.array(rlBids1), c='b')
    
    #ax.scatter(range(len(keys)), optTbs, np.array(optBids)-np.array(rlBids))
    plt.scatter(range(len(tbs0)), np.array(optBids0)-np.array(rlBids0), c='r')
    plt.scatter(range(len(tbs1)), np.array(optBids1)-np.array(rlBids1), c='b')
    
    probStr = (prob + " ") if len(prob)>0 else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('States')
    plt.ylabel('Bids made')
    
    #ax.plot_trisurf(rl.boards, rl.tiebreaker, rl.bids, antialiased=True) #linewidth=0.2, 
    #ax.plot_trisurf(opt.boards, opt.tiebreaker, opt.bids, antialiased=True)
    #ax.plot_trisurf(opt.boards, opt.tiebreaker, np.array(opt.bids)-np.array(rl.bids), antialiased=True)
    plt.show()

    
def calcOptStrat(totalChips):
    opt = Player("px", False, "", -1, totalChips)
    opt.generateStrategy(totalChips)
    fd = open('simDiscreteRich', 'wb')
    pickle.dump(opt.nodesToDiscreteRich, fd)
    fd.close()
    fm = open('simMoveBid', 'wb')
    pickle.dump(opt.nodesToMoveBid, fm)
    fm.close()

def testAgent(chips, prob, rlStrat):
    AverageErrorAndWins(chips, prob, rlStrat, 1, 20000) # training
    print("testing:")
    p1 = Player("agent", prob, rlStrat, 1, chips, learn_rate=0, exp_rate=0)
    p1.loadPolicy("sim_policy_p1")
    #AverageErrorAndWins(chips, prob, rlStrat, 1, 1, p1)


if __name__ == "__main__":
    rlStrat = "TD"
    chips = 8
    prob = ""
    #AverageError(chips, prob, rlStrat, 10, 20000)
    #opt = Player("p2", "TD", 1, chips)
    #optGame = Sim()
    #Plot(chips, rlStrat, opt, optGame, 20000)
    #calcOptStrat(chips)
    #gc.disable()
    #testAgent(chips, prob, rlStrat)
    AverageErrorAndWins(chips, prob, rlStrat, 1, 20000)
    #TwoRlAgents(chips, prob, rlStrat, "random", 1, 10000)
    #PlotWins(chips, prob, rlStrat, "optimal", 10, 20000)
    #b = Board([[(5, 3), (5, 4)], [(1, 0), (2, 1), (3, 2), (4, 3)]])
#    b.updateState((5,3), 1)
    #b.showBoard()
    #print(b.getHash())
    #b.showBoard()
    
    #sys.stdout = open("boards.txt", "w")
    #for b in getBoards(9):
    #    b.showBoard()
#    for b in getBoards(15):
#        myChildren = b.nextBoards(1)
#        oppChildren = b.nextBoards(-1)
#        for c in myChildren:
#            c.showBoard()
#        for c in oppChildren:
#            c.showBoard()
    #sys.stdout.close()
    #getBoards(9)
    #for prob in [True]:
    #   for strat in ["state-value1", "state-value2", "action-value1"]:
    #        AverageError(chips, prob, strat, 3, 20000)

