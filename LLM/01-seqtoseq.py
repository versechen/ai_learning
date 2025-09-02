import random

import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


# Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout,
                           batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)  # Add time dimension
        embedded = self.dropout(self.embedding(trg))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[:, 0]  # Start with <sos> token for first input
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


# Hyperparameters
INPUT_DIM = 10  # Size of source vocabulary
OUTPUT_DIM = 10  # Size of target vocabulary
ENC_EMB_DIM = 16  # Embedding dimension for encoder
DEC_EMB_DIM = 16  # Embedding dimension for decoder
HIDDEN_DIM = 32  # Hidden layer size
N_LAYERS = 2  # Number of LSTM layers
ENC_DROPOUT = 0.5  # Dropout for encoder
DEC_DROPOUT = 0.5  # Dropout for decoder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# Generate dummy data
def generate_dummy_data(batch_size, seq_len, vocab_size):
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    trg = torch.randint(1, vocab_size, (batch_size, seq_len))
    return src, trg


# Training parameters
BATCH_SIZE = 32
SEQ_LEN = 5
VOCAB_SIZE = 10
N_EPOCHS = 5
CLIP = 1

# Training loop
for epoch in range(N_EPOCHS):
    src, trg = generate_dummy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    iterator = [(src, trg)]  # Here, we're using one batch for simplicity
    train_loss = train(model, iterator, optimizer, criterion, CLIP)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
