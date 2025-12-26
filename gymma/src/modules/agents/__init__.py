from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_ppo_agent import RNNPPOAgent
from .n_rnn_agent import NRNNAgent
REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["n_rnn"] = NRNNAgent