import matplotlib.pyplot as plt

iterations = 151

with open("MLP.txt", "r") as f:
    lines = f.read().replace("[", "").split("]")
    train_losses = lines[0].split(",")
    test_losses = lines[1].split(",")
    train_acc = lines[2].split(",")
    test_acc = lines[3].split(",")

    for i in range(150):
        train_losses[i] = float(train_losses[i])
        test_losses[i] = float(test_losses[i])
        train_acc[i] = float(train_acc[i])
        test_acc[i] = float(test_acc[i])

with open("tree.txt", "r") as f:
    lines = f.read().replace("[", "").split("]")
    t_train_losses = lines[0].split(",")
    t_test_losses = lines[1].split(",")
    t_train_acc = lines[2].split(",")
    t_test_acc = lines[3].split(",")

    for i in range(150):
        t_train_losses[i] = float(t_train_losses[i])
        t_test_losses[i] = float(t_test_losses[i])
        t_train_acc[i] = float(t_train_acc[i])
        t_test_acc[i] = float(t_test_acc[i])

with open("multi_tree.txt", "r") as f:
    lines = f.read().replace("[", "").split("]")
    mt_train_losses = lines[0].split(",")
    mt_test_losses = lines[1].split(",")
    mt_train_acc = lines[2].split(",")
    mt_test_acc = lines[3].split(",")

    for i in range(150):
        mt_train_losses[i] = float(mt_train_losses[i])
        mt_test_losses[i] = float(mt_test_losses[i])
        mt_train_acc[i] = float(mt_train_acc[i])
        mt_test_acc[i] = float(mt_test_acc[i])

plt.subplot(121)
plt.plot(range(1, iterations), train_losses, 'b--', label='MLP_train_loss')
plt.plot(range(1, iterations), test_losses, 'b-', label='MLP_test_loss')
plt.plot(range(1, iterations), t_train_losses, 'r--', label='tree_train_loss')
plt.plot(range(1, iterations), t_test_losses, 'r-', label='tree_test_loss')
plt.plot(range(1, iterations), mt_train_losses, 'g--', label='multi_tree_train_loss')
plt.plot(range(1, iterations), mt_test_losses, 'g-', label='multi_tree_test_loss')
plt.title('loss')
plt.xlabel('iters')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(range(1, iterations), train_acc, 'b--', label='MLP_train_acc')
plt.plot(range(1, iterations), test_acc, 'b-', label='MLP_test_acc')
plt.plot(range(1, iterations), t_train_acc, 'r--', label='tree_train_acc')
plt.plot(range(1, iterations), t_test_acc, 'r-', label='tree_test_acc')
plt.plot(range(1, iterations), mt_train_acc, 'g--', label='multi_tree_train_acc')
plt.plot(range(1, iterations), mt_test_acc, 'g-', label='multi_tree_test_acc')
plt.title('acc')
plt.xlabel('iters')
plt.ylabel('acc')
plt.legend()

plt.show()
