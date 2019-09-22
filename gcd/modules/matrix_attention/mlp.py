import torch
from allennlp.modules.matrix_attention import MatrixAttention
from overrides import overrides


@MatrixAttention.register('mlp')
class MLPAttention(MatrixAttention):
    def __init__(self,
                 encoder_dim: int,
                 decoder_dim: int,
                 attention_dim: int) -> None:
        super().__init__()
        self.W = torch.nn.Linear(encoder_dim + decoder_dim, attention_dim)
        self.v = torch.nn.Linear(attention_dim, 1, bias=False)

    @overrides
    def forward(self,
                decoder_outputs: torch.Tensor,
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        num_decoder_tokens = decoder_outputs.size(1)
        num_encoder_tokens = encoder_outputs.size(1)

        decoder_outputs = decoder_outputs.unsqueeze(2)
        encoder_outputs = encoder_outputs.unsqueeze(1)

        decoder_outputs = decoder_outputs.expand(-1, -1, num_encoder_tokens, -1)
        encoder_outputs = encoder_outputs.expand(-1, num_decoder_tokens, -1, -1)

        concat = torch.cat([decoder_outputs, encoder_outputs], dim=-1)
        affinities = self.v(torch.tanh(self.W(concat))).squeeze(-1)
        return affinities
