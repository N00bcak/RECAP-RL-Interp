# RECAP Capstone Project: Attention Neuron

This one-week project examines the [Attention Neuron](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionNeuron), a peculiar architecture by [Tang and Ha](arxiv.org/abs/2109.02869) which claims to provide permutation invariant representations of the agent's observations.

The project examines pre-trained models from the project and attempts to understand how the Attention Neuron works, through evaluations, visualizations, and ablations.

## Features
- **Hooking**: The project attaches hooks to the Attention Neuron to read/write hidden states and attention scores.
- **Visualizations**: To understand the model's behavior, the project visualizes the attention scores and final representations.
- **Ablations**:
  - **LSTM Recurrent Statte**: The project evaluates the model's performance with and without the LSTM recurrent state.