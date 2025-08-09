import random
import threading
import time
import numpy as np
from chessEnv import ChessEnv
from game import Game
from agent import Agent
import argparse
import logging

# Improved logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

from GUI.display import GUI

class Main:
    def __init__(self, player: bool, local_predictions: bool = False, model_path: str = None):
        self.player = player
        logging.info(f"Initializing game. Player is {'White' if self.player else 'Black'}")
        
        self.opponent = Agent(local_predictions=local_predictions, model_path=model_path)

        if self.player:
            self.game = Game(ChessEnv(), None, self.opponent)
        else:
            self.game = Game(ChessEnv(), self.opponent, None)

        print("*" * 50)
        print(f"You play the {'white' if self.player else 'black'} pieces!")
        print("*" * 50)

        self.previous_moves = (None, None)
        self.GUI = GUI(800, 800, player)
        self.GUI.gameboard.board.fen = self.game.env.board.fen()

        thread = threading.Thread(target=self.play_game)
        thread.start()

        self.GUI_loop()

    def GUI_loop(self):
        while True:
            self.GUI.draw()

    def play_game(self):
        self.game.reset()
        winner = None
        while winner is None:
            if self.player == self.game.turn:
                self.get_player_move()
                self.game.turn = not self.game.turn
            else:
                logging.info("Opponent is thinking...")
                self.opponent_move()
                self.GUI.make_move(self.game.env.board.move_stack[-1])

            if self.game.env.board.is_game_over():
                winner = Game.get_winner(self.game.env.board.result(claim_draw=True))
                print("White wins" if winner == 1 else "Black wins" if winner == -1 else "Draw")

    def get_player_move(self):
        while True:
            time.sleep(0.2)
            try:
                if len(self.GUI.gameboard.board.move_stack) and not len(self.game.env.board.move_stack):
                    break
                if self.game.env.board.move_stack[-1] != self.GUI.gameboard.board.move_stack[-1]:
                    break
            except IndexError:
                continue
        logging.info("Player move received.")
        self.game.env.board.push(self.GUI.gameboard.board.move_stack[-1])

    def opponent_move(self):
        self.GUI.gameboard.selected_square = None
        self.previous_moves = self.game.play_move(stochastic=False, previous_moves=self.previous_moves, save_moves=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=str, default=None, choices=('white', 'black'))
    parser.add_argument('--local-predictions', action='store_true')
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    if args["local_predictions"] and args["model"] is None:
        print("When using local predictions, specify the path to the model to use.")
        exit(1)

    player = args['player'].lower().strip() == 'white' if args['player'] else np.random.choice([True, False])
    model_path = args["model"]
    local_predictions = args["local_predictions"]

    m = Main(player, local_predictions, model_path)
