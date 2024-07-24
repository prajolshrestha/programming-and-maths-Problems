#include <torch/torch.h>
#include <iostream>

/**
 * Implements a simple linear regression model using PyTorch.
 * The model has a single input feature and a single output.
 * The forward pass computes the output as a linear combination of the input and the model parameters (weight and bias).
 */
class LinearRegression : public torch::nn::Module {
    public:
    LinearRegression() {
        // initialize weights and bias
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
    // create dummy data
    auto x = torch::tensor({1.0,2.0,3.0,4.0,5.0});
    auto y = torch::tensor({2.0,4.0,6.0,8.0,10.0});

    // create a model, loos function and optimizer
    LinearRegression model;
    torch::optim::SGD optimizer(model.parameters(), /*lr=*/0.01);
    auto loss_fn = torch::nn::MSELoss();

    // Training loop 
    for (int epoch =0; epoch < 500; epoch++) {
        // Forward pass
        auto pred = model.forward(x);
        auto loss = loss_fn(pred, y);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // optimize
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Print final parameters
    std::cout << "Final weight: " << model.parameters()[0].item<float>() << std::endl;
    std::cout << "Final bias: " << model.parameters()[1].item<float>() << std::endl;
}

