# Neural Network Normalization Guide

## Types of Normalization

### Channel-based Normalizations
Normalize across channel dimensions, preserving channel independence.

1. **BatchNorm**
   - Normalizes: Across batch for each channel
   - Best for: CNNs with batch_size ≥ 32
   - Usage: `nn.BatchNorm2d(num_channels)`
   - Formula: `(x - batch_mean) / sqrt(batch_var + eps)`

2. **InstanceNorm**
   - Normalizes: Each channel independently per sample
   - Best for: Style transfer, image generation
   - Usage: `nn.InstanceNorm2d(num_channels)`
   - Formula: `(x - instance_mean) / sqrt(instance_var + eps)`

3. **GroupNorm**
   - Normalizes: Groups of channels
   - Best for: Small batch sizes in CNNs
   - Usage: `nn.GroupNorm(num_groups, num_channels)`
   - Formula: `(x - group_mean) / sqrt(group_var + eps)`

### Sample-based Normalizations
Normalize across entire feature representation per sample.

1. **LayerNorm**
   - Normalizes: All features for each sample
   - Best for: Transformers, NLP
   - Usage: `nn.LayerNorm(normalized_shape)`
   - Formula: `(x - mean) / sqrt(var + eps) * gamma + beta`

2. **RMSNorm**
   - Normalizes: Like LayerNorm but using RMS
   - Best for: Efficient transformer variants
   - Usage: Custom implementation
   - Formula: `x / sqrt(mean(x^2) + eps) * scale`

## Other Important Normalizations

1. **Weight Normalization**
   - Normalizes: Layer weights
   - Usage: `weight_norm(layer)`

2. **Spectral Normalization**
   - Normalizes: Weight matrices by spectral norm
   - Best for: GAN stability
   - Usage: `spectral_norm(layer)`

3. **Adaptive Instance Normalization (AdaIN)**
   - Best for: Style transfer
   - Combines: Content and style statistics

4. **Conditional Batch Normalization**
   - Best for: Class-conditional generation
   - Adds: Class-specific parameters

## Quick Decision Guide

```python
if task == "CNN_vision":
    if batch_size >= 32:
        use BatchNorm
    else:
        use GroupNorm
elif task == "transformers_or_NLP":
    use LayerNorm  # or RMSNorm for efficiency
elif task == "style_transfer":
    use InstanceNorm
elif batch_size == "very_small":
    use GroupNorm
```

```
BatchNorm:    [N] → [C] [H W]    # Normalizes across batch
InstanceNorm: N [C] → [H W]      # Normalizes each channel
GroupNorm:    N [G C] → [H W]    # Normalizes channel groups
LayerNorm:    N → [C H W]        # Normalizes all features
RMSNorm:      N → [C H W]        # Normalizes all features (RMS)
```
Where `[]` indicates dimensions normalized together.

## Key Points

- Each normalization has learnable parameters (usually gamma, beta)
- Choice depends on: batch size, task, architecture
- Can mix different normalizations in one model
- Helps with:
  - Training stability
  - Gradient flow
  - Internal covariate shift
  - Faster convergence