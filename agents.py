import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch import nn

from go_search_problem import GoProblem, GoState, Action, HeuristicGoProblem
from heuristic_go_problems import GoProblemSimpleHeuristic, GoProblemLearnedHeuristic
from models import ValueNetwork, load_model

MAXIMIZER = 0
MIMIZER = 1

class GameAgent(ABC):
    """Abstract base class for all Go game agents."""
    
    @abstractmethod
    def get_move(self, state: GoState, time_limit: float) -> Action:
        """Get the best move for the given state within the time limit.
        
        Args:
            state: Current game state
            time_limit: Maximum time in seconds to spend on this move
            
        Returns:
            Action index representing the chosen move
        """
        pass

    def reset(self):
        """Reset any internal state of the agent if necessary.
            Called after a new game is started.
        """
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)

#############################################
# 
#
# Part 1: Basic Adversarial Search Algorithms
#
#
#############################################

class MinimaxAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        MinimaxAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        search_problem = self.search_problem

        def minimax(state: GoState, depth: int) -> float:
            # Terminal state
            if search_problem.is_terminal_state(state):
                return search_problem.get_result(state)
            # Depth cutoff -> heuristic
            if depth == 0:
                return search_problem.heuristic(state, state.player_to_move())

            actions = search_problem.get_available_actions(state)
            if state.player_to_move() == MAXIMIZER:
                value = -float('inf')
                for a in actions:
                    ns = search_problem.transition(state, a)
                    v = minimax(ns, depth - 1)
                    if v > value:
                        value = v
                return value
            else:
                value = float('inf')
                for a in actions:
                    ns = search_problem.transition(state, a)
                    v = minimax(ns, depth - 1)
                    if v < value:
                        value = v
                return value

        # Choose best action from root
        best_action = None
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
            for a in search_problem.get_available_actions(game_state):
                ns = search_problem.transition(game_state, a)
                v = minimax(ns, self.depth - 1)
                if v > best_value:
                    best_value = v
                    best_action = a
        else:
            best_value = float('inf')
            for a in search_problem.get_available_actions(game_state):
                ns = search_problem.transition(game_state, a)
                v = minimax(ns, self.depth - 1)
                if v < best_value:
                    best_value = v
                    best_action = a

        return best_action

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        AlphaBetaAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        search_problem = self.search_problem

        def alphabeta(state: GoState, depth: int, alpha: float, beta: float) -> float:
            if search_problem.is_terminal_state(state):
                return search_problem.get_result(state)
            if depth == 0:
                return search_problem.heuristic(state, state.player_to_move())

            actions = search_problem.get_available_actions(state)
            if state.player_to_move() == MAXIMIZER:
                value = -float('inf')
                for a in actions:
                    ns = search_problem.transition(state, a)
                    v = alphabeta(ns, depth - 1, alpha, beta)
                    if v > value:
                        value = v
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = float('inf')
                for a in actions:
                    ns = search_problem.transition(state, a)
                    v = alphabeta(ns, depth - 1, alpha, beta)
                    if v < value:
                        value = v
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value

        # Root-level selection
        best_action = None
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            for a in search_problem.get_available_actions(game_state):
                ns = search_problem.transition(game_state, a)
                v = alphabeta(ns, self.depth - 1, alpha, beta)
                if v > best_value:
                    best_value = v
                    best_action = a
                alpha = max(alpha, best_value)
        else:
            best_value = float('inf')
            alpha = -float('inf')
            beta = float('inf')
            for a in search_problem.get_available_actions(game_state):
                ns = search_problem.transition(game_state, a)
                v = alphabeta(ns, self.depth - 1, alpha, beta)
                if v < best_value:
                    best_value = v
                    best_action = a
                beta = min(beta, best_value)

        return best_action

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)



def create_value_agent_from_model():
    """
    Create agent object from saved model. 
    This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.

    In the game_runner file, there is a factory function that will call this function to create an agent.
    You can run games with your agent against other agents by running game_runner.py with the appropriate command line arguments.
    """
    # DONE: Update model path to your saved model
    model_path = "value_model.pt"

    sample_problem = GoProblem()
    sample_state = sample_problem.start_state
    # Create a temporary heuristic object to use its encoding method
    temp_h = GoProblemLearnedHeuristic()
    feature_size = len(temp_h.encoding(sample_state))

    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    # DONE: Try with other heuristic agents (IDS/AB/Minimax)
    learned_agent = GreedyAgent(heuristic_search_problem)

    return learned_agent


################################################
#
# Part 2: Advanced Adversarial Search Algorithms
#
################################################

class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """

        # Time tracking
        start_time = time.time()

        # Safety cushion
        cushion = 0.2

        def alphabeta_with_timeout(state, depth, alpha, beta, start_time, time_limit):
            if time.time() - start_time >= time_limit - cushion:
                return self.search_problem.heuristic(state, state.player_to_move())
            if self.search_problem.is_terminal_state(state):
                return self.search_problem.get_result(state)
            if depth == 0:
                return self.search_problem.heuristic(state, state.player_to_move())

            actions = self.search_problem.get_available_actions(state)
            if state.player_to_move() == MAXIMIZER:
                value = -float('inf')
                for a in actions:
                    ns = self.search_problem.transition(state, a)
                    v = alphabeta_with_timeout(ns, depth - 1, alpha, beta, start_time, time_limit)
                    if v > value:
                        value = v
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:
                value = float('inf')
                for a in actions:
                    ns = self.search_problem.transition(state, a)
                    v = alphabeta_with_timeout(ns, depth - 1, alpha, beta, start_time, time_limit)
                    if v < value:
                        value = v
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value
        
        def search_at_depth(depth):
            best_action = None
            if game_state.player_to_move() == MAXIMIZER:
                best_value = -float('inf')
                alpha = -float('inf')
                beta = float('inf')
                for a in self.search_problem.get_available_actions(game_state):
                    # Check time limit
                    if time.time() - start_time >= time_limit - cushion:
                        break
                    ns = self.search_problem.transition(game_state, a)
                    v = alphabeta_with_timeout(ns, depth - 1, alpha, beta, start_time, time_limit)
                    if v > best_value:
                        best_value = v
                        best_action = a
                    alpha = max(alpha, best_value)
            else:
                best_value = float('inf')
                alpha = -float('inf')
                beta = float('inf')
                for a in self.search_problem.get_available_actions(game_state):
                    # Check time limit
                    if time.time() - start_time >= time_limit - cushion:
                        break
                    ns = self.search_problem.transition(game_state, a)
                    v = alphabeta_with_timeout(ns, depth - 1, alpha, beta, start_time, time_limit)
                    if v < best_value:
                        best_value = v
                        best_action = a
                    beta = min(beta, best_value)
            return best_action
    
        actions = self.search_problem.get_available_actions(game_state)
        best_action = actions[0] if actions else None
        depth = 1
        while time.time() - start_time < time_limit - cushion:
            action = search_at_depth(depth)
            if action is not None:
                best_action = action
                depth += 1
            else:
                break
        return best_action


            

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)
    

class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)
    
class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        start_time = time.time()
        cushion = 0.2

        # Create root node for MCTS
        root = MCTSNode(state=game_state, parent=None, action=None)

        # Main MCTS loop
        while time.time() - start_time < time_limit - cushion:
            # SELECT
            leaf = self._select(root)
            # EXPAND
            children = self._expand(leaf)
            for child in children:
                #SIMULATE
                result = self._simulate(child)
                # BACKPROPAGATE
                self._backpropagate(child, result)
        # Return action of child with most visist
        return self._best_child(root, c=0).action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not self.search_problem.is_terminal_state(current.state):
            # Check if current has unexpanded action s
            available_actions = self.search_problem.get_available_actions(current.state)
            expanded_actions = [child.action for child in current.children]

            # If leaf    
            if len(expanded_actions) < len(available_actions):
                return current
            
            # Choose best child
            current = self._best_child(current, self.c)

        return current
    
    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        if self.search_problem.is_terminal_state(node.state):
            return [node]
    
        # Get unexpanded actions
        available_action = self.search_problem.get_available_actions(node.state)
        expanded_actions = [child.action for child in node.children]
        unexpanded_actions = [action for action in available_action if action not in expanded_actions]

        # If no unexpanded, return node
        if not unexpanded_actions:
            return []
        
        # Create children for all unexpanded actions
        new_children = []
        for action in unexpanded_actions:
            # Create new state
            new_state = self.search_problem.transition(node.state, action)
            # Create child node
            child = MCTSNode(state=new_state, parent=node, action=action)
            new_children.append(child)
            node.children.append(child)
            new_children

        return new_children
    
    def _simulate(self, node: MCTSNode) -> float:
        current_state = node.state

        # Random play until game ends
        while not self.search_problem.is_terminal_state(current_state):
            actions = self.search_problem.get_available_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            current_state = self.search_problem.transition(current_state, action)
        
        # Get results (Black POV)
        result = self.search_problem.get_result(current_state)

        # 1 for BLACK wins, 0 for WHITE wins
        return 1.0 if result > 0 else 0.0

    def _backpropagate(self, node: MCTSNode, result: float):
        current = node

        while current is not None:
            current.visits += 1
            if current.parent is not None:
                parent_player = current.parent.state.player_to_move()
                if parent_player == 0:
                    current.value += result
                else:
                    current.value += 1 - result
            else:
                root_player = current.state.player_to_move()
                if root_player == 0:
                    current.value += result
                else:
                    current.value += 1 - result

            current = current.parent

    def _best_child(self, node: MCTSNode, c: float) -> MCTSNode:
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            if child.visits == 0:
                # Unvisited child (exploration)
                return child
            
            # UCT formula
            exploitation = child.value / child.visits
            exploration = c * np.sqrt(np.log(node.visits) / child.visits)
            uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child



    def __str__(self):
        return "MCTS"
    

###################################################
#
# Part 3: Final Agent
#
###################################################

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    return MCTSAgent()

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return None
