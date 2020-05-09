import os
from pathlib import Path


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, voc_id):
        self.base = Path(__file__).parent.parent.parent.expanduser().resolve()

        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = self.base/'logs_wavernn/checkpoints'
        self.voc_latest_weights = self.voc_checkpoints/'latest_weights.pyt'
        self.voc_latest_optim = self.voc_checkpoints/'latest_optim.pyt'

        self.voc_output = self.base/'logs_wavernn/model_outputs'
        self.voc_step = self.voc_checkpoints/'step.npy'
        self.voc_log = self.voc_checkpoints/'log.txt'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)


    def get_voc_named_weights(self, name):
        """Gets the path for the weights in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_weights.pyt'

    def get_voc_named_optim(self, name):
        """Gets the path for the optimizer state in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_optim.pyt'


