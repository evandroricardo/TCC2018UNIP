import loader
import rede

training_data, validation_data, test_data = loader.load_data_wrapper()
training_data = list(training_data)


net = rede.Network([2, 3, 1])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
