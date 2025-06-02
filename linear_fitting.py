import argparse
import matplotlib.animation as animation
import torch
from torch.utils.cpp_extension import load

ext = load(name="linear_fitting", sources=["linear_fitting.cu"])

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x)

        # Run CUDA kernel
        y = torch.empty_like(x)
        ext.linear_forward(x, a.item(), b.item(), y)
        ctx.a = a
        ctx.b = b
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_a = torch.zeros_like(ctx.a)
        grad_b = torch.zeros_like(ctx.b)

        ext.linear_backward(x, grad_output.contiguous(), grad_a, grad_b)
        return None, grad_a, grad_b  # Only a, b have gradients

def linear_model(x, a, b):
    return LinearFunction.apply(x, a, b)

def train(args):
    # Target data
    x = torch.linspace(-2, 2, args.num_points, device=args.device)
    y_gt = -0.3 * x + 2 + 0.5 * torch.randn(args.num_points, device=args.device)

    # Initialize parameters
    a = torch.tensor([0.0], device=args.device)
    b = torch.tensor([0.0], device=args.device)
    a.requires_grad = True
    b.requires_grad = True

    # Optimizer and loss function
    optimizer = torch.optim.SGD([a, b], lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    # Animation frames
    frames = []  # stores (iter, a.item(), b.item(), y_pred snapshot)

    for i in range(args.iterations):
        optimizer.zero_grad()

        y_pred = linear_model(x, a, b)
        loss = mse_loss(y_pred, y_gt)
        
        loss.backward()
        optimizer.step()

        print(f"Iteration {i:04d}, Loss: {loss.item():.4f}, a: {a.item():.4f}, b: {b.item():.4f}")

        if i % 5 == 0:
            frames.append((i, a.item(), b.item(), y_pred.detach().cpu().clone()))

    # Animation
    visualize(x, y_gt, frames)

def visualize(x, y_gt, frames):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Linear Fitting")
    ax.set_xlim(x.min().cpu(), x.max().cpu())
    ax.set_ylim(y_gt.min().cpu() - 1, y_gt.max().cpu() + 1)
    ax.plot(x.cpu(), y_gt.cpu(), label="Ground Truth", color="gray", linestyle="dashed")
    line_pred, = ax.plot([], [], label="Prediction", color="blue")
    text_iter = ax.text(0.05, 0.95, "", transform=ax.transAxes, ha="left", va="top")

    def update(frame):
        i, a_val, b_val, y_pred = frame
        line_pred.set_data(x.cpu(), y_pred)
        text_iter.set_text(f"Iter {i}, a={a_val:.2f}, b={b_val:.2f}")
        return line_pred, text_iter

    ani = animation.FuncAnimation(fig, update, frames, interval=100, blit=True)
    ani.save("vis_linear_fit.mp4", writer="ffmpeg", dpi=150)

def grad_test(x: float, a: float, b: float):
    x_tensor = torch.tensor([x], dtype=torch.float32, device=args.device)
    a_tensor = torch.tensor([a], dtype=torch.float32, device=args.device, requires_grad=True)
    b_tensor = torch.tensor([b], dtype=torch.float32, device=args.device, requires_grad=True)
    passed = torch.autograd.gradcheck(linear_model, (x_tensor, a_tensor, b_tensor), eps=1e-3, atol=1e-2)
    print("Gradcheck linear passed:", passed)

def parse_args():
    parser = argparse.ArgumentParser(description="Linear Fitting with custom CUDA")
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points to fit",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # add device to args
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_test(0.1, 0.3, 2.0)  # Test gradients
    train(args)
