from tqdm import tqdm
import time

def train_model(model, train_loader, optimizer, criterion, device, epochs):
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for protein_feat, tgt_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            protein_feat = protein_feat.to(device)
            tgt_seq = tgt_seq.to(device)

            optimizer.zero_grad()
            output = model(protein_feat, tgt_seq[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_seq[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s", flush=True)

    return loss_history
