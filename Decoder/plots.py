import matplotlib.pyplot as plt
import config

def plot_loss(loss_history, save_path=config.save_lossplot):
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
