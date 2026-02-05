from utils.preprocessing import load_cifar10, CLASS_NAMES

x_train, y_train, x_test, y_test = load_cifar10()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
print("Example label:", y_train[0])
print("Class names:", CLASS_NAMES)
