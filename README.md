This project was focused on developing a chess engine inspired by AlphaZero, designed to learn and improve through playing against itself rather than relying on fixed rules.
Instead of having every possible move calculated in advance, moves were predicted and positions were evaluated based on patterns the system had learned over time. Countless self-play games were carried out, and the best-performing version of the model was kept after each training cycle.

A setup was created that allowed many self-play games to be run at the same time, so that data could be collected more quickly. A pre-trained model was made available, though it was not trained extensively due to limited computing power, meaning better results could be achieved with further training.

Overall, the project was carried out to show how deep reinforcement learning could be applied to chess in a way that allowed the engine to adapt and think in a more human-like manner, with its skills being improved through experience rather than through predefined instructions.
