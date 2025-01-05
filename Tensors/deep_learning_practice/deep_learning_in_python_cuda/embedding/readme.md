# Understanding PyTorch nn.Embedding

A simple guide to understand how embedding layers work in PyTorch.

## Overview

An embedding layer maps discrete indices to dense vectors. It's like a lookup table where each index gets its own learnable vector representation.


## How It Works

1. **Embedding Table**
   - Each index (0-9) gets a vector of size 3
   - These vectors start random and are learned during training
   - Similar items end up with similar vectors

2. **Vector Lookup**
   - Input indices are replaced with their corresponding vectors
   - Input shape: (2,5) → Output shape: (2,5,3)
   - Each index becomes its 3D vector representation

3. **Training**
   - Vectors are updated through backpropagation
   - Only used vectors get gradient updates
   - Vectors learn to capture meaningful relationships


After training, similar items will have similar vector representations in this 3D space.

## Key Points

- Each item gets its own learnable vector
- Vectors start random but become meaningful through training
- The embedding space captures relationships between items
- Similar items cluster together in the embedding space

## Knowledge space or Semantic space
- Distance between vectors = Semantic Similarity
- Direction in this space = Relationship or meaningful concepts
- Clusters = Categories
- You can even do vector arithmetic in this space! 
(Famous example: King - Man + Woman ≈ Queen)