import chess
import numpy as np
import threading
import logging
from tqdm import tqdm
from graphviz import Digraph

from chessEnv import ChessEnv
from node import Node
from edge import Edge
import config
from mapper import Mapping
import utils

# ... top imports remain unchanged

class MCTS:
    def __init__(self, agent: "Agent", state: str = chess.STARTING_FEN, stochastic=False):
        self.root = Node(state=state)
        self.game_path: list[Edge] = []
        self.cur_board: chess.Board = None
        self.agent = agent
        self.stochastic = stochastic

    def run_simulations(self, n: int) -> None:
        logging.info("Starting %d MCTS simulations...", n)
        for i in tqdm(range(n), desc="Simulating"):
            self.game_path = []
            leaf = self.select_child(self.root)
            logging.debug("Selected leaf node.")
            leaf.N += 1
            leaf = self.expand(leaf)
            logging.debug("Expanded node.")
            leaf = self.backpropagate(leaf, leaf.value)
            logging.debug("Backpropagated result.")

    def select_child(self, node: Node) -> Node:
        while not node.is_leaf():
            if not node.edges:
                return node
            noise = [1 for _ in node.edges]
            if self.stochastic and node == self.root:
                noise = np.random.dirichlet([config.DIRICHLET_NOISE] * len(node.edges))
            best_score = -np.inf
            best_edge = None
            for i, edge in enumerate(node.edges):
                score = edge.upper_confidence_bound(noise[i])
                if score > best_score:
                    best_score = score
                    best_edge = edge
            if best_edge is None:
                raise Exception("No edge found during selection.")
            node = best_edge.output_node
            self.game_path.append(best_edge)
        return node

    def expand(self, leaf: Node) -> Node:
        board = chess.Board(leaf.state)
        actions = list(board.generate_legal_moves())

        if not actions:
            outcome = board.outcome(claim_draw=True)
            leaf.value = 0 if outcome is None else (1 if outcome.winner == chess.WHITE else -1)
            return leaf

        input_state = ChessEnv.state_to_input(leaf.state)
        p, v = self.agent.predict(input_state)
        leaf.value = v

        probs = self.probabilities_to_actions(p, leaf.state)
        for action in actions:
            new_state = leaf.step(action)
            leaf.add_child(Node(new_state), action, probs.get(action.uci(), 0))
        return leaf

    def backpropagate(self, end_node: Node, value: float) -> Node:
        for edge in reversed(self.game_path):
            edge.input_node.N += 1
            edge.N += 1
            edge.W += value
            value = -value
        return end_node

    def probabilities_to_actions(self, probabilities: list, board: str) -> dict:
        probabilities = probabilities.reshape(config.amount_of_planes, config.n, config.n)
        self.cur_board = chess.Board(board)
        valid_moves = self.cur_board.generate_legal_moves()

        actions = {}
        for move in valid_moves:
            try:
                from_square = move.from_square
                piece = self.cur_board.piece_at(from_square)
                if not piece:
                    continue
                if move.promotion and move.promotion != chess.QUEEN:
                    piece_type, direction = Mapping.get_underpromotion_move(move.promotion, from_square, move.to_square)
                    plane_index = Mapping.mapper[piece_type][1 - direction]
                elif piece.piece_type == chess.KNIGHT:
                    direction = Mapping.get_knight_move(from_square, move.to_square)
                    plane_index = Mapping.mapper[direction]
                else:
                    direction, distance = Mapping.get_queenlike_move(from_square, move.to_square)
                    plane_index = Mapping.mapper[direction][abs(distance) - 1]

                row = from_square % 8
                col = 7 - (from_square // 8)
                actions[move.uci()] = probabilities[plane_index][col][row]
            except Exception as e:
                logging.warning(f"Failed to map move {move.uci()}: {e}")
        return actions
