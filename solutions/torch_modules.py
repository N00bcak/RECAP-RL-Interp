import numpy as np
import torch
import torch.nn as nn


def pos_table(n, dim):
    """Create a table of positional encodings."""

    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / dim)

    def get_angle_vec(x):
        return [get_angle(x, j) for j in range(dim)]

    tab = np.array([get_angle_vec(i) for i in range(n)]).astype(float)
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])
    return tab


class AttentionMatrix(nn.Module):
    """Self-attention matrix."""

    def __init__(self, dim_in_q, dim_in_k, msg_dim, bias=True, scale=True):
        super(AttentionMatrix, self).__init__()
        self.proj_q = nn.Linear(
            in_features=dim_in_q, out_features=msg_dim, bias=bias)
        self.proj_k = nn.Linear(
            in_features=dim_in_k, out_features=msg_dim, bias=bias)
        if scale:
            self.msg_dim = msg_dim
        else:
            self.msg_dim = 1

    def forward(self, data_q, data_k):
        q = self.proj_q(data_q)
        k = self.proj_k(data_k)
        if data_q.ndim == data_k.ndim == 2:
            dot = torch.matmul(q, k.T)
        else:
            dot = torch.bmm(q, k.permute(0, 2, 1))
        return torch.div(dot, np.sqrt(self.msg_dim))


class SelfAttentionMatrix(AttentionMatrix):
    """Self-attention matrix."""

    def __init__(self, dim_in, msg_dim, bias=True, scale=True):
        super(SelfAttentionMatrix, self).__init__(
            dim_in_q=dim_in,
            dim_in_k=dim_in,
            msg_dim=msg_dim,
            bias=bias,
            scale=scale,
        )


class AttentionLayer(nn.Module):
    """The attention mechanism."""

    def __init__(self, dim_in_q, dim_in_k, dim_in_v, msg_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.attention_matrix = AttentionMatrix(
            dim_in_q=dim_in_q,
            dim_in_k=dim_in_k,
            msg_dim=msg_dim,
        )
        self.proj_v = nn.Linear(in_features=dim_in_v, out_features=out_dim)
        self.mostly_attended_entries = None

    def forward(self, data_q, data_k, data_v):
        a = torch.softmax(
            self.attention_matrix(data_q=data_q, data_k=data_k), dim=-1)
        self.mostly_attended_entries = set(torch.argmax(a, dim=-1).numpy())
        v = self.proj_v(data_v)
        return torch.matmul(a, v)


class AttentionNeuronLayer(nn.Module):
    """Permutation invariant layer."""

    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 bias=True,
                 scale=True):
        super(AttentionNeuronLayer, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.pos_embedding = torch.from_numpy(
            pos_table(self.hidden_dim, self.pos_em_dim)
        ).float()
        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size=1 + self.act_dim, hidden_size=pos_em_dim)
        self.attention = SelfAttentionMatrix(
            dim_in=pos_em_dim, msg_dim=self.msg_dim, bias=bias, scale=scale)

    def forward(self, obs, prev_act):
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs.copy()).float().unsqueeze(-1)
        else:
            x = obs.unsqueeze(-1)
        obs_dim = x.shape[0]

        x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)
        if self.hx is None:
            self.hx = (
                torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
                torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
            )
        self.hx = self.lstm(x_aug, self.hx)

        w = torch.tanh(self.attention(
            data_q=self.pos_embedding.to(x.device), data_k=self.hx[0]))
        output = torch.matmul(w, x)
        return torch.tanh(output)

    def reset(self):
        self.hx = None

class HookedAttentionNeuronLayer(AttentionNeuronLayer):
    """
    Hooked attention neuron layer for debugging.
    This layer exposes the internal state and allows for write-then-read access to the activations.
    In other words, you can edit the results of the forward pass before they
    are cached in the activations dictionary.
    """

    def __init__(self, hook_fns = None, **kwargs):
        super(HookedAttentionNeuronLayer, self).__init__(**kwargs)
        self.hook_fns = hook_fns if hook_fns is not None else {}

    def forward(self, obs, prev_act):
        activations_dict = {}
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs.copy()).float().unsqueeze(-1)
        else:
            x = obs.unsqueeze(-1)
        obs_dim = x.shape[0]

        # Edit and cache the current observation.
        x = self.hook_fns.get('obs', lambda x: x)(x)
        activations_dict['obs'] = x

        # Edit and cache the previous action.
        prev_act = self.hook_fns.get('prev_act', lambda x: x)(prev_act)
        activations_dict['prev_act'] = prev_act

        x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)

        # Edit and cache the augmented input.
        x_aug = self.hook_fns.get('x_aug', lambda x: x)(x_aug)
        activations_dict['x_aug'] = x_aug
        
        if self.hx is None:
            self.hx = (
                torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
                torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
            )

        # Edit and cache the previous hidden state.
        # For ease of debugging, we express the hidden state as a dictionary
        # instead of a tuple.
        self.hx = self.hook_fns.get('hx_prev', lambda x: x)(self.hx)
        activations_dict['hx_prev'] = {'hidden': self.hx[0], 'cell': self.hx[1]}
        self.hx = self.lstm(x_aug, self.hx)

        # Edit and cache the current hidden state.
        # Again, we express the hidden state as a dictionary.
        self.hx = self.hook_fns.get('hx', lambda x: x)(self.hx)
        activations_dict['hx'] = {'hidden': self.hx[0], 'cell': self.hx[1]}

        # Edit and cache the positional embedding.
        # This is actually used as the query vector in the attention mechanism.
        # You can modify it to check the meaning of each feature.
        self.pos_embedding = self.hook_fns.get('pos_embedding', lambda x: x)(self.pos_embedding)
        activations_dict['pos_embedding'] = self.pos_embedding.to(x.device)
        
        # Compute the attention matrix.
        # Somewhat atypically, this attention mechanism uses Tanh instead of Softmax.        
        w = torch.tanh(self.attention(
            data_q=self.pos_embedding.to(x.device), data_k=self.hx[0]))
        
        # Edit and cache the attention matrix.
        w = self.hook_fns.get('attention_matrix', lambda x: x)(w)
        activations_dict['attention_matrix'] = w

        # Compute the output.
        output = torch.matmul(w, x)
        output = self.hook_fns.get('output', lambda x: x)(output)
        activations_dict['output'] = torch.tanh(output)

        return torch.tanh(output), activations_dict

class VisionAttentionNeuronLayer(nn.Module):
    """Permutation invariant layer for vision tasks."""

    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 patch_size=6,
                 stack_k=4,
                 with_learnable_ln_params=False,
                 stack_dim_first=False):
        super(VisionAttentionNeuronLayer, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.patch_size = patch_size
        self.stack_k = stack_k
        self.stack_dim_first = stack_dim_first
        self.pos_em_dim = pos_em_dim
        self.pos_embedding = torch.from_numpy(
            pos_table(self.hidden_dim, self.pos_em_dim)
        ).float()
        self.attention = AttentionLayer(
            dim_in_q=self.pos_em_dim,
            dim_in_k=(self.stack_k - 1) * self.patch_size**2 + self.act_dim,
            dim_in_v=self.stack_k * self.patch_size**2,
            msg_dim=self.msg_dim,
            out_dim=self.msg_dim,
        )
        # The normalization layers have no learnable parameters.
        self.input_ln = nn.LayerNorm(
            normalized_shape=self.patch_size**2,
            elementwise_affine=with_learnable_ln_params,
        )
        self.input_ln.eval()
        self.output_ln = nn.LayerNorm(
            normalized_shape=self.msg_dim,
            elementwise_affine=with_learnable_ln_params,
        )
        self.output_ln.eval()

    def get_patches(self, x):
        h, w, c = x.size()
        patches = x.unfold(
            0, self.patch_size, self.patch_size).permute(0, 3, 1, 2)
        patches = patches.unfold(
            2, self.patch_size, self.patch_size).permute(0, 2, 1, 4, 3)
        return patches.reshape((-1, self.patch_size, self.patch_size, c))

    def forward(self, obs, prev_act):
        if isinstance(obs, dict):
            # Puzzle pong may drop some patches.
            patch_to_keep_ix = obs['patches_to_use']
            obs = obs['obs']
        else:
            patch_to_keep_ix = None

        k, h, w = obs.shape
        assert k == self.stack_k
        if patch_to_keep_ix is None:
            num_patches = (h // self.patch_size) * (w // self.patch_size)
        else:
            num_patches = patch_to_keep_ix.size

        # AttentionNeuron is the first layer, so obs is numpy array.
        x_obs = torch.div(torch.from_numpy(obs).float(), 255.)

        # Create Key.
        x_k = torch.diff(x_obs, dim=0).permute(1, 2, 0)
        x_k = self.get_patches(x_k)
        if patch_to_keep_ix is not None:
            x_k = x_k[patch_to_keep_ix]
        assert x_k.shape == (
            num_patches, self.patch_size, self.patch_size, self.stack_k - 1)
        if self.stack_dim_first:
            x_k = x_k.permute(0, 3, 1, 2)
        x_k = torch.cat([
            torch.flatten(x_k, start_dim=1),
            torch.repeat_interleave(prev_act, repeats=num_patches, dim=0)
        ], dim=-1)

        # Create Value.
        x_v = self.get_patches(x_obs.permute(1, 2, 0)).permute(0, 3, 1, 2)
        if patch_to_keep_ix is not None:
            x_v = x_v[patch_to_keep_ix]
        x_v = self.input_ln(torch.flatten(x_v, start_dim=2))

        x = self.attention(
            data_q=self.pos_embedding,
            data_k=x_k,
            data_v=x_v.reshape(num_patches, -1),
        )
        return self.output_ln(torch.relu(x))
