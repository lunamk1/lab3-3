import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
#     labels = input_ids.clone()
#     probability_matrix = torch.full(labels.shape, mlm_prob)
#     special_tokens_mask = (input_ids == pad_token_id)
#     probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels[~masked_indices] = -100
#     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#     input_ids[indices_replaced] = mask_token_id
#     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#     random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
#     input_ids[indices_random] = random_words[indices_random]
#     return input_ids, labels

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    labels = input_ids.clone()

    # create probability matrix on same device as input_ids
    probability_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
    # put special tokens mask on same device
    special_tokens_mask = (input_ids == pad_token_id).to(input_ids.device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time: replace masked tokens with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    # 10% of the time: replace masked tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5, device=input_ids.device)
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        vocab_size, labels.shape, dtype=torch.long, device=input_ids.device
    )
    input_ids[indices_random] = random_words[indices_random]

    # 10% keep original token (handled automatically by not changing input_ids)

    return input_ids, labels


def train_bert(model, dataloader, tokenizer, val_split = 0.1, epochs=3, lr=5e-4, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    # ===========================
    # Split the dataloader's dataset into train and validation
    # ===========================
    dataset = dataloader.dataset
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    train_losses = []
    val_losses = []


    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)


            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.bool()

            masked_input_ids, labels = mask_tokens(
                input_ids,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                mlm_prob=0.15
            )
            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            hidden_states = model(masked_input_ids, token_type_ids, attention_mask)
            logits = model.mlm_head(hidden_states)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # ===== Validation =====
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                if attention_mask.ndim == 3:
                    attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.bool()

                masked_input_ids, labels = mask_tokens(
                    input_ids,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    mlm_prob=0.15
                )
                masked_input_ids = masked_input_ids.to(device)
                labels = labels.to(device)

                hidden_states = model(masked_input_ids, token_type_ids, attention_mask)
                logits = model.mlm_head(hidden_states)

                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = loss_fn(logits, labels)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

    # plt.plot(range(1, epochs+1), train_losses, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.title('Training Loss Curve')
    # plt.grid()
    # plt.show()
