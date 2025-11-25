import torch
import torch.nn as nn


class decoder_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(decoder_model, self).__init__()
        self.decoder = nn.GRU(input_size, hidden_size, num_layers,
                              bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, n):
        """
        x: (batch, 1, input_size) — начальный токен
        n: количество генерируемых элементов
        return: (batch, n, 1)
        """
        outputs = []
        #print("x in forward:", x.shape)
        cur_x = x  # (B, 1, input_size)
        hidden = None

        for _ in range(n):
            # прогоняем текущий вход через BiGRU (seq_len == 1)
            out, hidden = self.decoder(cur_x, hidden)
            #print("dec_out shape:", out.shape)# dec_out: (B, 1, hidden_size*2)
            out_step = self.output_layer(out[:, -1, :])
            outputs.append(out_step.unsqueeze(1))
            # сохраняем (B,1)

        # outputs: list длины n, каждый элемент (B,1) -> склеим в (B, n, 1)
        return torch.cat(outputs, dim=1)  # (B, n, 1)

