REGISTRY = {}
from .n_controller import NMAC
from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .lica_controller import LICAMAC
from .ppo_controller import PPOMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["n_mac"] = NMAC