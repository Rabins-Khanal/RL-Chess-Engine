import logging
import socket
import json
import numpy as np
import time
import os
import chess

from dotenv import load_dotenv
from mcts import MCTS
import config
from chessEnv import ChessEnv
import utils

load_dotenv()

class Agent:
    def __init__(self, local_predictions: bool = False, model_path=None, state=chess.STARTING_FEN):
        self.local_predictions = local_predictions

        if self.local_predictions:
            logging.info("[Agent] Using local prediction with model: %s", model_path)
            from tensorflow.python.ops.numpy_ops import np_config
            from tensorflow.keras.models import load_model
            np_config.enable_numpy_behavior()
            self.model = load_model(model_path)
        else:
            logging.info("[Agent] Connecting to server for prediction...")
            try:
                self.socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_to_server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                server = os.environ.get("SOCKET_HOST", "localhost")
                port = int(os.environ.get("SOCKET_PORT", 5001))
                self.socket_to_server.connect((server, port))
                logging.info("[Agent] Connected to prediction server at %s:%s", server, port)
            except Exception as e:
                logging.error("Failed to connect to prediction server: %s", e)
                exit(1)

        self.mcts = MCTS(self, state=state)

    def run_simulations(self, n: int = 1):
        logging.info("Running %d MCTS simulations...", n)
        self.mcts.run_simulations(n)

    def predict(self, data):
        if self.local_predictions:
            logging.debug("Predicting locally...")
            import local_prediction
            p, v = local_prediction.predict_local(self.model, data)
            return p.numpy(), v[0][0]
        else:
            return self.predict_server(data)

    def predict_server(self, data: np.ndarray):
        try:
            logging.debug("Sending input to prediction server...")
            self.socket_to_server.send(f"{len(data.flatten()):010d}".encode('ascii'))
            self.socket_to_server.send(data)

            logging.debug("Waiting for prediction response...")
            data_length = self.socket_to_server.recv(10)
            data_length = int(data_length.decode("ascii"))

            response = utils.recvall(self.socket_to_server, data_length)
            response = response.decode("ascii")
            response = json.loads(response)

            logging.debug("Prediction received.")
            return np.array(response["prediction"]), response["value"]
        except Exception as e:
            logging.error("Prediction server error: %s", e)
            raise
