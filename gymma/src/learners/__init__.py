from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .policy_gradient_v2 import PGLearner_v2
from .fmac_learner import FMACLearner
from .nq_learner import NQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner