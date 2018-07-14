from rede import Network

network = Network([2, 3, 1])
network.update_mini_batch([(0.5, 0.7)], 0.005)
network.sgd
