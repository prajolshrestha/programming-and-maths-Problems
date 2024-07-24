#include <torch/torch.h>
#include <iostream>

// class linear_regression is inheriting from torch::nn::Module
class LinearRegression : public torch::nn::Module { 
public:
    LinearRegression() {
        // Initialize the weight and bias
        weight = register_parameter("weight", torch::randn({1}));
        bias = register_parameter("bias", torch::randn({1}));
    }

    torch::Tensor forward(torch::Tensor x) {
        return weight * x + bias;
    }

private:
    torch::Tensor weight, bias;
};

int main() {
    // Create dummy data
    auto x = torch::linspace(0, 10, 100);
    auto y = 2 * x + 1 + torch::randn({100}) * 0.1;

    // Create model, loss function, and optimizer
    LinearRegression model;
    torch::optim::SGD optimizer(model.parameters(), /*lr=*/0.01);
    auto loss_fn = torch::nn::MSELoss();

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        auto pred = model.forward(x);
        auto loss = loss_fn(pred, y);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Print final parameters
    std::cout << "Final weight: " << model.parameters()[0].item<float>() << std::endl;
    std::cout << "Final bias: " << model.parameters()[1].item<float>() << std::endl;

    return 0;
}