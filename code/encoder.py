import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BERT-STYLE ENCODER MODEL ---

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        """
        Multi-Head Self-Attention layer.

        Args:
            hidden_size (int): Hidden size of the model.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (Tensor, optional): Attention mask of shape (batch_size, seq_len) or (batch_size, 1, seq_len).

        Returns:
            Tensor: Output after applying multi-head attention.
        """
        if mask is not None:
            while mask.ndim > 2:
                mask = mask.squeeze(1)

            key_padding_mask = ~mask.bool()
    
        else:
            key_padding_mask = None

        return self.attn(x, x, x, key_padding_mask=key_padding_mask)[0]


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        Feed-Forward Network.

        Args:
            hidden_size (int): Hidden size of the model.
            intermediate_size (int): Intermediate size of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for feed-forward network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after feed-forward operations.
        """
        return self.fc2(self.activation(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        """
        Single Transformer block with attention and feed-forward layers.

        Args:
            hidden_size (int): Hidden size.
            num_heads (int): Number of attention heads.
            intermediate_size (int): Size of intermediate feed-forward layer.
        """
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        """
        Forward pass for the transformer block.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor.
        """
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=512):
        """
        Encoder model: stacks multiple Transformer blocks.

        Args:
            vocab_size (int): Vocabulary size.
            hidden_size (int, optional): Hidden size. Defaults to 256.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_layers (int, optional): Number of transformer layers. Defaults to 4.
            intermediate_size (int, optional): Intermediate size in feed-forward. Defaults to 512.
            max_len (int, optional): Maximum input sequence length. Defaults to 512.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.type_emb = nn.Embedding(2, hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Forward pass for the encoder.

        Args:
            input_ids (Tensor): Input token IDs (batch_size, seq_len).
            token_type_ids (Tensor): Segment token IDs (batch_size, seq_len).
            attention_mask (Tensor): Attention mask (batch_size, seq_len).

        Returns:
            Tensor: Final hidden states (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_emb(input_ids) + self.pos_emb(positions) + self.type_emb(token_type_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        return x  # Return hidden states [batch_size, seq_len, hidden_size]
