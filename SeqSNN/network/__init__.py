from .base import NETWORKS

from .ann.tsrnn import TSRNN
from .ann.itransformer import ITransformer
from .ann.tcn2d import TemporalConvNet2D, TemporalBlock2D

from .snn.snn import TSSNN, TSSNN2D, ITSSNN2D
from .snn.spike_tcn import SpikeTemporalConvNet2D, SpikeTemporalBlock2D
from .snn.ispikformer import iSpikformer
from .snn.spikformer import Spikformer
from .snn.spikformer_CPG import SpikformerCPG
from .snn.spikernn import SpikeRNN, SpikeRNN2D

from .snn.spikingformer import Spikingformer
from .snn.spikingformer_xnor import Spikingformer_XNOR
from .snn.spikingformer_xnor_gray import Spikingformer_XNOR_Gray
from .snn.spikingformer_xnor_log import Spikingformer_XNOR_Log

from .snn.spikformer_xnor import Spikformer_XNOR
from .snn.spikformer_xnor_log import Spikformer_XNOR_Log
from .snn.spikformer_xnor_gray import Spikformer_XNOR_Gray

from .snn.qkformer import QKFormer
from .snn.qkformer_xnor_log import QKFormer_XNOR_Log
from .snn.qkformer_xnor_gray import QKFormer_XNOR_Gray
