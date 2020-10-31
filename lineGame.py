import numpy as np
import pickle
import random
import math
import sys
import matplotlib.pyplot as plt



def getBoards(d, n):
# =============================================================================
# 	Returns a list of boards with distance d from a finished game)
#This is useful because we must calculate Richman and discrete-Richman
# 	values backwards-recursively (that is, the Richman value of any state
# 	is determined by the Richman values of its child states). With the
#   states partitioned by distance from a full state, and with the knowledge
#   that any child states of a given gamestate must be 1 closer to a full
#   state, we can systematically calculate the Richman values for all
#   terminal nodes (that is, all states in distance0), then for all states
#   in distance1, then in distance2, and so on.
# =============================================================================
    b1 = Board(n)
    b1.token = 1+d
    b2 = Board(n)
    b2.token = n-d
    return [b1, b2]


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
        

    def __init__(self, n):
        self.token = math.floor(n/2) + 1
        self.length = n
        
    # get unique hash of current board state
    def getHash(self):
        return str(self.token)+str(self.length)
    
    def winner(self):
        if self.token == 1:     #token on left end
            return 1
        elif self.token == self.length:        #token on right end
            return -1
        # else game not ended yet
        return None
    
    def distance(self, symbol):
        if symbol == 1:
            return (self.token - 1)
        else:
            return (self.length - self.token)
    
    def getBoard(self):
        return (self.token, self.length)
    
    def copy(self):
        copy = Board(self.length)
        copy.token = self.token
        return copy
    
    def nextBoards(self, symbol):
        if (self.winner() is not None):
            return []
        c = self.copy()
        c.token += -1*symbol
        return [c]

    def updateState(self, symbol):
        self.token -= symbol
        
    def showBoard(self):
        # token: x  empty: o
        print()
        for i in range(1, self.token):
            print('-', end="")
        print("x", end="")
        for i in range(self.token + 1, self.length + 1):
            print('-', end="")
        print()
    
    def standardString(self):
        return str(self.token)
    
    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return (self.token==other.token and self.length==other.length)



class State(StateI):

    def __init__(self, n, p1, p2, totalChips):
        self.board = Board(n)
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        agentChips = math.ceil(0.5*totalChips)
        #agentChips =  math.ceil(0.5*totalChips)
        self.chips = [totalChips, agentChips, totalChips-agentChips]
        self.tieBreaker = 1
        
    def copy(self):
        p1 = self.p1
        p2 = self.p2
        copy = State(n, p1, p2, self.chips[0])
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
        return str(self.boardHash) + str(self.chips[1]) + str(self.chips[2]) + str(self.tieBreaker)

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
        self.board = Board(self.board.length)
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        agentChips = math.ceil(0.5*self.chips[0])
        self.chips = [self.chips[0], agentChips, self.chips[0]-agentChips]
        
    def bid(self): 
        p1Bid = min(self.p1.getBid(self, 1, 2, self.chips[1]), self.chips[1])
        p2Bid = min(self.p2.getBid(self, 2, 1, self.chips[2]), self.chips[2])
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
        self.chips[1] = int(self.chips[1] - p1Bid  + p2Bid)
        self.chips[2] = int(self.chips[2] - p2Bid + p1Bid)
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
        
        
    def win(self):
        w = self.board.winner()
        if not w==None:
            self.isEnd = True
        return w
# =============================================================================
#             if np.random.uniform(0, 1) <= 0.5:
#                 self.playerSymbol = 1
#                 return 1 
#             else:
#                 self.playerSymbol = -1
#                 return -1
# =============================================================================
            
            


class Player(AgentI):
    def __init__(self, name, prob, biddingStrategy, symbol, totalChips, n, exp_rate=0.5):
        self.name = name
        self.states = []  # record all positions taken
        self.bids = []  # record all bids taken
        self.lr = 0.6
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.biddingStrategy = biddingStrategy
        self.symbol = symbol
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
        self.nodesToDiscreteRich = {}

		# nodesToMoveBid is a list of dictionaries that hold
		# entries of the form
		# gameNode : (optimalMove, optimalBid),
		#
		# where optimalMove is of the form (row, col). The nodes 
		# are partitioned by their distance away from a full state
        self.nodesToMoveBid = {}

        if biddingStrategy == "optimal":
            self.generateStrategy(totalChips)
            
            
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

#        boards0 = getBoards(0, n)
#        for board in boards0:
#            if board.winner()==self.symbol:
#                self.nodesToDiscreteRich[0][board.getHash()] = 0.0
#            else:
#                self.nodesToDiscreteRich[0][board.getHash()] = totalChips + 1.0        
#                
#        boards0 = getBoards(math.floor(n/2), n)
#        for board in boards0:
#            if board.winner()==self.symbol:
#                self.nodesToDiscreteRich[0][board.getHash()] = 0.0
#            else:
#                self.nodesToDiscreteRich[0][board.getHash()] = totalChips + 1.0        


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

        for i in range(0,n):
			# Get all nodes that are i steps away from a full state
            nodes = getBoards(i, n)

            for node in nodes:

				# If the node is a win state for the agent, assign it 
				# a discrete-Richman value of 0.
                if node.winner()==self.symbol:
                    self.nodesToDiscreteRich[node.getHash()] = 0.0
                    continue
				# Else if the node is a win state for the opponent, assign
				# a discrete-Richman value of k+1.
                elif node.winner()==(-1*self.symbol):
                    self.nodesToDiscreteRich[node.getHash()] = totalChips + 1.0
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
                    #print("rich", self.nodesToDiscreteRich[myChild.getHash()])
                    if not(self.nodesToDiscreteRich.get(myChild.getHash()) == None) and Fmin > self.nodesToDiscreteRich[myChild.getHash()]:
                        Fmin = self.nodesToDiscreteRich[myChild.getHash()]
                for oppChild in oppChildren:
                    if not(self.nodesToDiscreteRich.get(oppChild.getHash()) == None) and Fmax < self.nodesToDiscreteRich[oppChild.getHash()]:
                        Fmax = self.nodesToDiscreteRich[oppChild.getHash()]

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
                #print("max", FmaxVal)
                #print("min", FminVal)
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
                    bid = max(0,abs(FmaxVal-FminVal)/2.0 - 0.75)

				# Else (i.e., if Fsum is even and Fmin \in \N)
                else:
                    epsilon = 0.0
                    bid = abs(FmaxVal-FminVal)/2.0
                
                self.nodesToDiscreteRich[node.getHash()] = math.floor(Fsum/2.0) + epsilon
                self.nodesToMoveBid[node.getHash()] = bid 
                #print(bid)


    def randomBid(self, availableTokens, tieBreaker, symbol):
        tb = 0
        if (tieBreaker==symbol):  #have tiebreaker
            if np.random.uniform(0, 1) <= 0.5:
                tb = 1
        return random.randint(0, availableTokens) + 0.25*tb
    
    def getProb(self, stateHash, b):
        (bidsWon, bidsNum) = self.stateProbs.get((stateHash, b, 1), (0,1))
        return (bidsWon + 0) / (bidsNum + 0)
    
    def addStateBidProb(self, state, bid, win, tb):
        (won, num) = self.stateProbs.get((state, bid, tb), (0,0))
        self.stateProbs[state, bid, tb] = (won+win, num+1)
    
    def stateValBid(self, prob, pId, oId, availableTokens, tieBreaker, symbol, state):
        bid = 0
        tb = 0
        value_max = -999
        for b in range(int(availableTokens) + 1):
            if (tieBreaker==symbol):  #have tiebreaker
                next_state = state.copy()
                next_state.tieBreaker *= -1
                next_state.board.updateState(symbol)
                next_stateHash = next_state.getHash()
                value = self.states_value.get((next_stateHash), 0)
                if prob:
                    bidWinProb = self.getProb(state.getHash(), b)
                    value *= bidWinProb
                if value >= value_max:
                    value_max = value
                    bid = b
                    tb = 1
            next_state = state.copy()
            next_state.chips[pId] = min(next_state.chips[pId] - 1, 0)
            next_state.chips[oId] = max(next_state.chips[oId] + 1, next_state.chips[0])
            next_state.board.updateState(symbol)
            next_stateHash = next_state.getHash()
            value = self.states_value.get((next_stateHash), 0)
            if prob:
                bidWinProb = self.getProb(state.getHash(), b)
                value *= bidWinProb
            if value >= value_max:
                value_max = value
                bid = b
                tb = 0
        return (bid, tb)
    
    def actionValBid(self, prob, pId, oId, availableTokens, tieBreaker, symbol, state):
        bid = 0
        tb = 0
        value_max = -999
        for b in range(int(availableTokens) + 1):
            if (tieBreaker==symbol):  #have tiebreaker
                next_state = state.copy()
                next_state.tieBreaker *= -1
                next_state.board.updateState(symbol)
                next_stateHash = next_state.getHash()
                value = self.states_value.get((next_stateHash, b), 0)
                if prob:
                    bidWinProb = self.getProb(state.getHash(), b)
                    value *= bidWinProb
                if value >= value_max:
                    value_max = value
                    bid = b
                    tb = 1
            next_state = state.copy()
            next_state.chips[pId] -= 1
            next_state.chips[oId] += 1
            next_state.board.updateState(symbol)
            next_stateHash = next_state.getHash()
            if prob:
                    bidWinProb = self.getProb(state.getHash(), b)
                    value *= bidWinProb
            value = self.states_value.get((next_stateHash, b), 0)
            if value >= value_max:
                value_max = value
                bid = b
                tb = 0
        return (bid, tb)
    
    def getBid(self, state, pId, oId, availableTokens):
        prob = self.prob
        if (self.biddingStrategy == "random"):
            # do random bid
            tb = 0
            if (state.tieBreaker==self.symbol):  #have tiebreaker
                if np.random.uniform(0, 1) <= 0.5:
                    tb = 1
            bid = random.randint(0, availableTokens)
            self.data[(state.board.standardString(), tb)] = bid
            return bid + 0.25*tb
        elif (self.biddingStrategy == "state-value1" or self.biddingStrategy == "TD"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                return self.randomBid(availableTokens, state.tieBreaker, self.symbol)
            else:
                # do greedy bid
                (bid, tb) = self.stateValBid(prob, pId, oId, availableTokens, state.tieBreaker, self.symbol, state)
                self.data[(state.board.standardString(), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "state-value2"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                return self.randomBid(availableTokens, state.tieBreaker, self.symbol)
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
                        if prob:
                            bidWinProb = self.getProb(state.getHash(), b)
                            value *= bidWinProb
                        if value >= value_max:
                            value_max = value
                            bid = b
                            tb = 1
                    value = self.states_value.get((stateHash), 0)
                    if prob:
                        bidWinProb = self.getProb(state.getHash(), b)
                        value *= bidWinProb
                    if value >= value_max:
                        value_max = value
                        bid = b
                        tb = 0
                self.data[(state.board.standardString(), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "action-value1"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                return self.randomBid(availableTokens, state.tieBreaker, self.symbol)

            else:
                # do greedy bid
                (bid, tb) = self.stateValBid(prob, pId, oId, availableTokens, state.tieBreaker, self.symbol, state)
                self.data[(state.board.standardString(), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "action-value2"):
            if np.random.uniform(0, 1) <= self.exp_rate:
                # do random bid
                return self.randomBid(availableTokens, state.tieBreaker, self.symbol)
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
                        if prob:
                            bidWinProb = self.getProb(state.getHash(), b)
                            value *= bidWinProb
                        if value >= value_max:
                            value_max = value
                            bid = b
                            tb = 1
                    value = self.states_value.get((stateHash), 0)
                    if prob:
                        bidWinProb = self.getProb(state.getHash(), b)
                        value *= bidWinProb
                    if value >= value_max:
                        value_max = value
                        bid = b
                        tb = 0
                self.data[(state.board.standardString(), tb)] = bid
                return (bid + tb*0.25)
        elif (self.biddingStrategy == "optimal"):
            numBlanks = state.board.distance(self.symbol)
            bid = self.nodesToMoveBid[state.board.getHash()]
            tb = 0
            if state.tieBreaker==self.symbol and bid % 1 == 0.25:
                tb = 1
            self.data[(state.board.standardString(), tb)] = bid
            return round(bid)
                    
        
    # append a hash state
    def addState(self, state):
        self.states.append(state)
        
       # append a bid
    def addBid(self, bid):
        self.bids.append(bid)
        
    # after each round, backpropagate and update state values
    def update(self, state, next_state, me):
        if self.biddingStrategy == "TD":
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
            self.states_value[st_hash] += self.lr * (self.states_value[next_st_hash] - self.states_value[st_hash])
            #state.board.showBoard()
            #print(self.states_value[st_hash])
        

    # at the end of game, backpropagate and update state value
    def feedReward(self, reward):
        if self.biddingStrategy == "action-value1" or self.biddingStrategy == "action-value2":
            for st, b in zip(reversed(self.states), reversed(self.bids)):
                if self.states_value.get((st, b)) is None:
                    self.states_value[(st, b)] = 0.5
                self.states_value[(st, b)] += self.lr * (self.decay_gamma * reward - self.states_value[(st, b)])
                reward = self.states_value[(st, b)]
                #elif not self.biddingStrategy == "TD":
        else:
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0.5
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
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
        pass

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass
    
class BiddingLine(GameI):
    
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
                self.wins.append((self.p1Win)/(i-1))
            if i % 1000 == 0:
                state.p1.exp_rate *= state.p1.decay_gamma
                state.p2.exp_rate *= state.p2.decay_gamma
                print("Rounds {}".format(i))
                #print(sum(state.p1.states_value.values()))

            while not state.isEnd:
                old_state = state.copy()
                turn = state.bid()
                if turn == 1:
                    # Player 1
                    #upate board state
                    state_hash = state.getHash()
                    state.p1.addState(state_hash)
                    state.p2.addState(state_hash)
                    
                    state.board.updateState(state.p1.symbol)
                    
                    state.p1.update(old_state, state, state.p1.symbol)
                    state.p2.update(old_state, state, state.p2.symbol)
                    
                    
                    # check board status if it is end
    
                    win = state.win()
                    if win is not None:
                        # self.showBoard()
                        # ended with p1 either win or draw
                        self.p1Win += 1
                        state.giveReward()
                        state.p1.reset()
                        state.p2.reset()
                        state.reset()
                        break
    
                else:
                    # Player 2
                    state_hash = state.getHash()
                    state.p2.addState(state_hash)
                    state.p1.addState(state_hash)
                    
                    state.board.updateState(state.p2.symbol)
                    
                    state.p1.update(old_state, state, state.p1.symbol)
                    state.p2.update(old_state, state, state.p2.symbol)
                    

                    win = state.win()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.p2Win += 1
                        state.giveReward()
                        state.p1.reset()
                        state.p2.reset()
                        state.reset()
                        break

    # play with human
    def play2(self, state):
        while not state.isEnd:
            turn = state.bid()
            if turn == 1:
                # Player 1
                state.board.updateState(1)
                print("Human: " + str(state.chips[2]) + "   Computer: " + str(state.chips[1]))
                if (state.tieBreaker == 1):
                    print("Computer has the tiebreaker.")
                else:
                    print("Human has the tiebreaker.")
                state.board.showBoard()
                # check board status if it is end
                win = state.board.winner()
                if win is not None:
                    if win == 1:
                        print(state.p1.name, "wins!")
                    else:
                        print("tie!")
                    state.reset()
                    break

            else:
                # Player 2
                state.board.updateState(-1)
                print("Human: " + str(state.chips[2]) + "   Computer: " + str(state.chips[1]))
                if (state.tieBreaker == 1):
                    print("Computer has the tiebreaker.")
                else:
                    print("Human has the tiebreaker.")
                state.board.showBoard()
                win = state.board.winner()
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
    n = 5
    prob = False

    p1 = Player("p1", prob, biddingStrategy, -1, chips, n)
    p2 = Player("p2", prob, biddingStrategy, 1, chips, n)

    st = State(n, p1, p2, chips)
    game = BiddingLine()
    print("training...")
    game.play(st, 50000)

    # play with human
    p1 = Player("computer", prob, biddingStrategy, 1, chips, n, exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(n, p1, p2, chips)
    game.play2(st)
    
def Optimal():
    # training
    biddingStrategy = "optimal"
    chips = 8
    n = 5
    prob = False

    game = BiddingLine()
    
    # play with human
    p1 = Player("computer", prob, biddingStrategy, 1, chips, n, exp_rate=0)
    p2 = HumanPlayer("human")

    st = State(n, p1, p2, chips)
    game.play2(st)
    
    
def Plot(n, chips, prob, rlStrat, opt, optGame, rounds):
    rl = Player("p1", prob, rlStrat, 1, chips, n)
    
    rlSt = State(n, rl, opt, chips)
    rlGame = BiddingLine()
    print("training...")
    rlGame.play(rlSt, rounds)
    #PlotStrats(prob, rlStrat, rl, opt)
    return PlotError3(prob, rlStrat, rlGame, opt, n)
        
def Wins(n, chips, prob, rlStrat, opt, optGame, rounds):
    rl = Player("p1", prob, rlStrat, 1, chips, n)
    
    rlSt = State(n, rl, opt, chips)
    rlGame = BiddingLine()
    print("training...")
    rlGame.play(rlSt, rounds)
    print(rlGame.p1Win)
    print(rlGame.p2Win)
    print(rl.states_value)
    #PlotStrats(prob, rlStrat, rl, opt)
    return PlotWin(prob, rlStrat, rlGame, opt, n)

def boardHeuristic(value):
     (key, bid) = value
     (token, tb) = key
     return int(token)

def AverageError(n, chips, prob, rlStrat, trials, rounds):
    opt = Player("p2", prob, "optimal", -1, chips, n)
    optGame = BiddingLine()
    
    errors = []
    for i in range(trials):
        errors.append(Plot(n, chips, prob, rlStrat, opt, optGame, rounds))
    print(sum(errors)/len(errors))

def AverageError2(n, chips, prob, strat1, strat2, trials, rounds):
    p2 = Player("p2", prob, strat2, -1, chips, n)
    optGame = BiddingLine()
    
    errors = []
    for i in range(trials):
        errors.append(Plot(n, chips, prob, rlStrat, p2, optGame, rounds))
    print(sum(errors)/len(errors))
    
def PlotWins(n, chips, prob, strat1, strat2, trials, rounds):
    p2 = Player("p2", prob, strat2, -1, chips, n)
    optGame = BiddingLine()
    
    wins = []
    for i in range(trials):
        wins.append(Wins(n, chips, prob, rlStrat, p2, optGame, rounds))
    print(sum(wins)/len(wins))


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
    probStr = "Probability " if prob else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]


def PlotError2(prob, rlStrat, rlGame, opt, n):
    rlDicts = rlGame.dicts
    errors = []
    for i in range(len(rlDicts)):
        rDict = rlDicts[i]
        meanError = 0.0
        keys = rDict.keys()
        rDict = {key: rDict[key] for key in keys}
        (rKeys, rBids) = map(list, zip(*rDict.items()))
        for (boardStr, rlTb) in rKeys:
            board = Board(n)
            board.token = int(boardStr)
            optBid = opt.nodesToMoveBid[board.getHash()]
            meanError += abs(float(rDict[(boardStr, rlTb)]) - float(optBid))
        errors.append(meanError / len(rKeys))
    
    plt.scatter(range(len(errors)), errors)
    # Add title and axis names
    probStr = "Probability " if prob else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]
        

def PlotError3(prob, rlStrat, rlGame, opt, n):
    rlDicts = rlGame.p1Dicts
    optDicts = rlGame.p2Dicts
    errors = []
    for i in range(len(rlDicts)):
        rDict = rlDicts[i]
        oDict = optDicts[i]
        meanError = 0.0
        keys = rDict.keys() & oDict.keys()
        if len(keys)==0:
            continue
        rDict = {key: rDict[key] for key in keys}
        oDict = {key: oDict[key] for key in keys}
        (rKeys, rBids) = map(list, zip(*rDict.items()))
        for (boardStr, rlTb) in rKeys:
            board = Board(n)
            board.token = int(boardStr)
            meanError += abs(float(rDict[(boardStr, rlTb)]) - float(oDict[(boardStr, rlTb)]))
        errors.append(meanError / len(rKeys))
    
    plt.scatter(range(len(errors)), errors)
    # Add title and axis names
    probStr = "Probability " if prob else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean difference in bids made')
    plt.show()
    return errors[-1]

def PlotWin(prob, rlStrat, rlGame, opt, n):
    wins = rlGame.wins
    
    plt.scatter(range(len(wins)), wins)
    # Add title and axis names
    probStr = "Probability " if prob else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('Number of rounds trained (in 100s)')
    plt.ylabel('Mean winning rate')
    plt.show()
    return wins[-1]
        

    
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
    
    probStr = "Probability " if prob else ""
    plt.title(probStr + rlStrat + " vs. Optimal Strategy")
    plt.xlabel('States')
    plt.ylabel('Bids made')
    
    #ax.plot_trisurf(rl.boards, rl.tiebreaker, rl.bids, antialiased=True) #linewidth=0.2, 
    #ax.plot_trisurf(opt.boards, opt.tiebreaker, opt.bids, antialiased=True)
    #ax.plot_trisurf(opt.boards, opt.tiebreaker, np.array(opt.bids)-np.array(rl.bids), antialiased=True)
    plt.show()



if __name__ == "__main__":
    rlStrat = "TD"
    chips = 8
    n = 5
    prob = False
    #opt = Player("p2", prob, "state-value1", 1, chips, n)
    #optGame = BiddingLine()
    #Plot(n, chips, prob, "state-value1", opt, optGame, 200)
    #opt = Player("p2", "TD", 1, chips)
    #optGame = BiddingTicTacToe()
    #Plot(chips, rlStrat, opt, optGame, 20000)
    
    #AverageError2(n, chips, prob, rlStrat, "random", 5, 20000)
    PlotWins(n, chips, prob, rlStrat, "random", 2, 50000)
    #for prob in [True, False]:
    #    for strat in ["state-value1", "state-value2", "action-value1", "action-value2", "TD"]:
    #        AverageError(n, chips, prob, strat, 3, 4000)

