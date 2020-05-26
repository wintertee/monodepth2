from networks.PackNeSt_decoder import PackNeSt_decoder
from networks.PackNeSt_encoder import PackNeSt_encoder
from networks.resnet_encoder import ResnetEncoder
from networks.PackNet01 import PackNet01
# from networks.bottleneck import Bottleneck

from torchsummary import summary

encoder = PackNeSt_encoder()
# encoder = ResnetEncoder(18,False)
# decoder = PackNeSt_decoder()
# bottleneck = Bottleneck(64,64,radix=1)
# encoder = PackNet01()

# summary(bottleneck,(64,50,50))

summary(encoder,(3,224,224))
