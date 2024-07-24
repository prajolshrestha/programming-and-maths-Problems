/**
 * Defines a simple linear regression model using PyTorch.
 * The model takes an input tensor of size `in_dim` and produces an output tensor of size `out_dim`.
 * The model is trained using L1 loss and stochastic gradient descent with momentum.
 * The training process runs for 10,000 epochs and prints the loss every 1,000 epochs.
 * After training, the final predictions are printed.
 */
#include <torch/torch.h>
#include <iostream>

class LinearRegression : public torch::nn::Module {
public:
    LinearRegression(int64_t in_dim, int64_t out_dim) {
        linear = register_module("linear", torch::nn::Linear(in_dim, out_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }

private:
    torch::nn::Linear linear{nullptr};
};

int main() {
    // Check if CUDA is available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }

    // Set random seed
    torch::manual_seed(42);

    // Inputs and targets
    auto inps = torch::rand({3}).to(device);
    auto tgts = torch::tensor({1.0, 2.0}).to(device);

    // Create model and move it to the appropriate device
    auto linear_model = std::make_shared<LinearRegression>(3, 2);
    linear_model->to(device);

    // Optimizer
    torch::optim::SGD optimizer(linear_model->parameters(), torch::optim::SGDOptions(1e-5).momentum(0.9));

    std::cout << "Starting training ..." << std::endl;

    // Training
    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 1. Initialize gradients
        optimizer.zero_grad();

        // 2. Forward pass
        auto outputs = linear_model->forward(inps.unsqueeze(0));

        // 3. Compute loss
        auto loss = torch::l1_loss(outputs.squeeze(), tgts);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss.item<float>() << std::endl;
        }

        // 4. Backward pass
        loss.backward();

        // 5. Update parameters
        optimizer.step();
    }

    // Evaluation
    linear_model->eval();
    torch::NoGradGuard no_grad;
    auto preds = linear_model->forward(inps.unsqueeze(0));
    std::cout << "Final predictions: " << preds << std::endl;

    return 0;
}