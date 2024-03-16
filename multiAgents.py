# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, PriorityQueue
from game import Directions
import random
import util
from game import Agent
import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == "Stop":
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if tempDistance > distance:
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="betterEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        # code lại minmax vì code của thầy em thấy nó không được chính xác nhất
        def minmax(state, agentIdx, depth):
            # Nếu tới độ sâu  tối đa hoặc đến trạng thái kết thúc
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # Nếu tới số đối tượng tối đa và đạt tới độ sâu tối đa thì trả về giá trị
            if agentIdx == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    # nếu chưa đạt tới độ sâu tối đa thì  quay lại pacman hành động
                    return minmax(state, 0, depth + 1)
            else:
                # MAX pacman
                if agentIdx == 0:
                    maxEva = -math.inf
                    # Lấy tất cả trạng thái của pacman
                    for action in state.getLegalActions(agentIdx):
                        tamp = minmax(
                            state.generateSuccessor(agentIdx, action),
                            agentIdx + 1,
                            depth + 1,
                        )
                        # Lấy bước đi cá giá trị lớn nhất
                        maxEva = max(maxEva, tamp)
                    return maxEva
                # MIN ghost
                else:
                    minEva = math.inf
                    for action in state.getLegalActions(agentIdx):
                        tamp = minmax(
                            state.generateSuccessor(agentIdx, action),
                            agentIdx + 1,
                            depth + 1,
                        )
                        # Lấy bước đi cá giá trị nhỏ nhất
                        minEva = min(minEva, tamp)
                    return minEva

        # result = max(gameState.getLegalActions(0), key=lambda x: minmax(
        #     gameState.generateSuccessor(0, x), 1, 1))
        # lấy action max của pacman và cập nhật lịch sử di chuyển của pacman
        result = "Stop"
        maxaction = -math.inf
        for action in gameState.getLegalActions(0):
            value = minmax(gameState.generateSuccessor(0, action), 1, 1)
            if maxaction < value:
                maxaction = value
                result = action
            elif maxaction == value:
                if random.random() < 0.5:
                    result = action
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def minmaxalphabeta(state, agentIdx, depth, alpha, beta):
            # Nếu tới độ sâu  tối đa hoặc đến trạng thái kết thúc
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # Nếu tới số đối tượng tối đa và đạt tới độ sâu tối đa thì trả về giá trị
            if agentIdx == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minmaxalphabeta(state, 0, depth + 1, alpha, beta)
            else:
                # MAX pacman
                if agentIdx == 0:
                    maxEva = -math.inf
                    # Lấy tất cả trạng thái của pacman
                    for action in state.getLegalActions(agentIdx):
                        tamp = minmaxalphabeta(
                            state.generateSuccessor(agentIdx, action),
                            agentIdx + 1,
                            depth + 1,
                            alpha,
                            beta,
                        )
                        # Lấy bước đi cá giá trị lớn nhất
                        maxEva = max(maxEva, tamp)
                        # cắt tỉa Alpha
                        alpha = max(alpha, maxEva)
                        if beta <= alpha:
                            break
                    return maxEva
                # MIN ghost
                else:
                    minEva = math.inf
                    for action in state.getLegalActions(agentIdx):
                        tamp = minmaxalphabeta(
                            state.generateSuccessor(agentIdx, action),
                            agentIdx + 1,
                            depth + 1,
                            alpha,
                            beta,
                        )
                        minEva = min(minEva, tamp)
                        # cắt tỉa beta
                        beta = min(beta, minEva)
                        if beta <= alpha:
                            break
                    return minEva

        # result = max(gameState.getLegalActions(0), key=lambda x: minmaxalphabeta(
        #     gameState.generateSuccessor(0, x), 1, 1, -math.inf, math.inf))
        # lấy action max của pacman và cập nhật lịch sử di chuyển của pacman
        maxaction = -math.inf
        result = "Stop"
        for action in gameState.getLegalActions(0):
            value = minmaxalphabeta(
                gameState.generateSuccessor(0, action), 1, 1, -math.inf, math.inf
            )
            if maxaction < value:
                maxaction = value
                result = action
            elif maxaction == value:
                if random.random() < 0.5:
                    result = action
        return result


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()
        def expectimax(state, agentIdx, depth):
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # Nếu tới số đối tượng tối đa và đạt tới độ sâu tối đa thì trả về giá trị
            if agentIdx == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    # nếu chưa đạt tới độ sâu tối đa thì  quay lại pacman hành động
                    return expectimax(state, 0, depth + 1)
            else:
                # MAX pacman
                if agentIdx == 0:
                    maxEva = -math.inf
                    # Lấy tất cả trạng thái của pacman
                    for action in state.getLegalActions(agentIdx):
                        tamp = expectimax(
                            state.generateSuccessor(agentIdx, action),
                            agentIdx + 1,
                            depth + 1,
                        )
                        # Lấy bước đi cá giá trị lớn nhất
                        maxEva = max(maxEva, tamp)
                    return maxEva
                # MIN ghost
                else:
                    weight = 0
                    actions = state.getLegalActions(agentIdx)
                    step = len(actions)
                    if step > 0:
                        for action in actions:
                            tamp = expectimax(
                                state.generateSuccessor(agentIdx, action),
                                agentIdx + 1,
                                depth + 1,
                            )
                            # Lấy tổng trọng số
                            weight += tamp
                        # vì yêu cầu của thầy là xác suất như nhau nên mỗi bước có tỷ lệ 1/số bước đi
                        return weight / step

        # result = max(gameState.getLegalActions(0), key=lambda x: expectimax(
        #     gameState.generateSuccessor(0, x), 1, 1))
        # lấy action max của pacman và cập nhật lịch sử di chuyển của pacman
        result = "Stop"
        maxaction = -math.inf
        for action in gameState.getLegalActions(0):
            value = expectimax(gameState.generateSuccessor(0, action), 1, 1)
            if maxaction < value:
                maxaction = value
                result = action
            elif maxaction == value:
                if random.random() < 0.5:
                    result = action
        return result


score_direc = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
    "Stop": (0, 0),
}


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Hàm tím kiếm A*
    def aStarSearch(start, end, walls, heuristic=manhattanDistance):
        fringe = PriorityQueue()
        start_heuristic = heuristic(start, end)
        visited_nodes = set()
        fringe.push((start, 0), start_heuristic)
        while not fringe.isEmpty():
            get_xy, get_cost = fringe.pop()
            if get_xy == end:
                return get_cost
            if not get_xy in visited_nodes:
                visited_nodes.add(get_xy)
                for action in score_direc.values():
                    nextation = (get_xy[0] + action[0], get_xy[1] + action[1])
                    if nextation not in walls and not nextation in visited_nodes:
                        get_heuristic = heuristic(nextation, end)
                        fringe.push((nextation, get_cost + 1), get_cost + get_heuristic)
        return 0

    # util.raiseNotDefined()
    # Lấy vị trí của tường
    walls = currentGameState.getWalls()
    setwalls = set(walls.asList())
    # Lấy vị trí của pacman
    newPos = currentGameState.getPacmanPosition()
    # Lấy vị trí của thức ăn
    newFood = currentGameState.getFood()
    # Lấy trạng thái của ghost
    newGhostStates = currentGameState.getGhostStates()
    # Vị trí cục năng lượng
    newCapsules = currentGameState.getCapsules()
    # Thời gian đếm ngược bị ăn của ma
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # Điểm trò chơi
    game_score = currentGameState.getScore()
    foodList = newFood.asList()
    openGhosts = []
    closesGhosts = []
    # Lấy ghost pacman ăn được và không ăn được
    for i in range(len(newScaredTimes)):
        if newScaredTimes[i] > 0:
            openGhosts.append([newGhostStates[i], newScaredTimes[i]])
        else:
            closesGhosts.append(newGhostStates[i])
    # Đánh giá khi pacman ăn được ghosts
    score = 0
    if openGhosts != []:
        #  khoảng cách / thời gian còn lại của ghost được ăn
        openGhost = min(
            [
                manhattanDistance(newPos, ghost[0].getPosition()) / ghost[1] * 10
                for ghost in openGhosts
            ]
        )
        score += 9000 / (openGhost + 1)
    # Đánh giá khi ghost lại càng gần
    if closesGhosts:
        closestGhost = min(
            [manhattanDistance(newPos, ghost.getPosition()) for ghost in closesGhosts]
        )
        if closestGhost < 2:
            score -= 10000000
        else:
            score += -70 / (closestGhost + 1)
    # đánh giá khi ăn thức ăn
    if foodList:
        closestFood = min([aStarSearch(newPos, food, setwalls) for food in foodList])
        score += 60 / (closestFood + 1)
    # Đánh giá khi ăn power
    if newCapsules:
        closestCapsule = min(
            [aStarSearch(newPos, caps, setwalls) for caps in newCapsules]
        )
        score += 90 / (closestCapsule + 1)
    # Tổng điểm đánh giá
    return score + game_score * 50


# Abbreviation
better = betterEvaluationFunction
