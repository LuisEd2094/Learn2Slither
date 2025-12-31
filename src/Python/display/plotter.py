import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, title: str = "Training Progress"):
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10))
        self.fig.suptitle(title)

        self._init_length_subplot()
        self._init_steps_subplot()
        self._init_loss_subplot()
        self._init_learning_rate_axis()

        plt.tight_layout()

    def _init_length_subplot(self):
        """Initialize snake length subplot with max and average length lines."""
        self.ax1.set_xlabel("Game Number")
        self.ax1.set_ylabel("Snake Length")
        self.ax1.grid(True, alpha=0.3)
        self.maxlens = []
        self.avglens = []
        (self.line_maxlen,) = self.ax1.plot(
            [], [], label="Max Length", color="tab:orange", linewidth=2
        )
        (self.line_avglen,) = self.ax1.plot(
            [],
            [],
            label="Avg Length",
            color="tab:blue",
            linewidth=2,
            alpha=0.7,
        )
        self.text_avglen = self.ax1.text(
            0.98,
            0.95,
            "",
            transform=self.ax1.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        self.ax1.legend(loc="upper left")

    def _init_steps_subplot(self):
        """Initialize steps subplot with step count and average steps lines."""
        self.ax2.set_xlabel("Game Number")
        self.ax2.set_ylabel("Steps")
        self.ax2.grid(True, alpha=0.3)
        self.steps = []
        self.avgsteps = []
        (self.line_steps,) = self.ax2.plot(
            [], [], label="Steps", color="tab:green", linewidth=2
        )
        (self.line_avgsteps,) = self.ax2.plot(
            [],
            [],
            label="Avg Steps",
            color="tab:purple",
            linewidth=2,
            alpha=0.7,
        )
        self.text_avgsteps = self.ax2.text(
            0.98,
            0.95,
            "",
            transform=self.ax2.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        self.ax2.legend(loc="upper left")

    def _init_loss_subplot(self):
        """Initialize loss subplot with loss and average loss lines."""
        self.ax3.set_xlabel("Game Number")
        self.ax3.set_ylabel("Loss", color="tab:red")
        self.ax3.tick_params(axis='y', labelcolor="tab:red")
        self.ax3.grid(True, alpha=0.3)
        self.losses = []
        self.avgloss = []
        (self.line_loss,) = self.ax3.plot(
            [], [], label="Loss", color="tab:red", linewidth=1, alpha=0.7
        )
        (self.line_avgloss,) = self.ax3.plot(
            [], [], label="Avg Loss", color="tab:brown", linewidth=2
        )
        self.text_avgloss = self.ax3.text(
            0.98,
            0.95,
            "",
            transform=self.ax3.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )
        self.ax3.legend(loc="upper left")

    def _init_learning_rate_axis(self):
        """Initialize secondary y-axis for learning rate visualization."""
        self.ax3_lr = self.ax3.twinx()
        self.ax3_lr.set_ylabel("Learning Rate", color="tab:cyan")
        self.ax3_lr.tick_params(axis='y', labelcolor="tab:cyan")
        self.learning_rates = []
        (self.line_lr,) = self.ax3_lr.plot(
            [],
            [],
            label="Learning Rate",
            color="tab:cyan",
            linewidth=2,
            linestyle="--",
        )
        self.ax3_lr.legend(loc="upper right")

    def update(
        self,
        max_len: int,
        steps: int,
        loss: float = None,
        lr: float = None,
    ):
        self._update_length_data(max_len)
        self._update_steps_data(steps)
        self._update_loss_data(loss)
        self._update_learning_rate_data(lr)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _update_length_data(self, max_len: int):
        """Update snake length metrics and visualization."""
        self.maxlens.append(max_len)
        window = min(100, len(self.maxlens))
        avg_len = np.mean(self.maxlens[-window:])
        self.avglens.append(avg_len)

        x = list(range(1, len(self.maxlens) + 1))
        self.line_maxlen.set_data(x, self.maxlens)
        self.line_avglen.set_data(x, self.avglens)
        self.text_avglen.set_text(f"Avg Len: {avg_len:.2f}")
        self.ax1.relim()
        self.ax1.autoscale_view()

    def _update_steps_data(self, steps: int):
        """Update steps metrics and visualization."""
        self.steps.append(steps)
        window = min(100, len(self.steps))
        avg_steps = np.mean(self.steps[-window:])
        self.avgsteps.append(avg_steps)

        x = list(range(1, len(self.steps) + 1))
        self.line_steps.set_data(x, self.steps)
        self.line_avgsteps.set_data(x, self.avgsteps)
        self.text_avgsteps.set_text(f"Avg Steps: {avg_steps:.1f}")
        self.ax2.relim()
        self.ax2.autoscale_view()

    def _update_loss_data(self, loss: float):
        """Update loss metrics and visualization."""
        if loss is not None:
            self.losses.append(loss)
            window = min(100, len(self.losses))
            avg_loss = np.mean(self.losses[-window:])
            self.avgloss.append(avg_loss)

            x = list(range(1, len(self.losses) + 1))
            self.line_loss.set_data(x, self.losses)
            self.line_avgloss.set_data(x, self.avgloss)
            self.text_avgloss.set_text(f"Avg Loss: {avg_loss:.4f}")
            self.ax3.relim()
            self.ax3.autoscale_view()

    def _update_learning_rate_data(self, lr: float):
        """Update learning rate visualization."""
        if lr is not None:
            self.learning_rates.append(lr)
            x = list(range(1, len(self.learning_rates) + 1))
            self.line_lr.set_data(x, self.learning_rates)
            self.ax3_lr.relim()
            self.ax3_lr.autoscale_view()
