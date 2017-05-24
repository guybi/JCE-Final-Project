
from networks.simple_net import build_simple_cnn14
from networks.double_net import build_double_cnn14
from networks.triple_net import build_triple_cnn14


def build_network(network_type, x, weights, biases):
    if network_type == 'simple':
        pred = build_simple_cnn14(x, weights, biases)
    if network_type == 'double':
        pred = build_double_cnn14(x, weights, biases)
    if network_type == 'triple':
        pred = build_triple_cnn14(x, weights, biases)

    return pred
