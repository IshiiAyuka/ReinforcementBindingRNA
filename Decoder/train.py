from tqdm import tqdm
import time
import config

def train_model_NAR(model, train_loader, optimizer, criterion, device, epochs):
    loss_history = []
    pad_id = config.rna_vocab_NAR["<pad>"]

    for epoch in range(epochs):
        model.train()
        total_loss_sum = 0.0       
        total_token_count = 0  
        start_time = time.time()

        for protein_feat, tgt_seq, _  in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            protein_feat = protein_feat.to(device)
            tgt_seq = tgt_seq.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(protein_feat, tgt_seq[:, :-1])
            labels = tgt_seq[:, 1:]

            loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            valid_tokens = (labels != pad_id).sum().item()
            total_loss_sum += loss.item()
            total_token_count += valid_tokens

        avg_loss = total_loss_sum / max(1, total_token_count)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s", flush=True)

    return loss_history

def train_model_AR(model, train_loader, optimizer, criterion, device, epochs):
    loss_history = []
    pad_id = config.rna_vocab["<pad>"]

    for epoch in range(epochs):
        model.train()
        total_loss_sum = 0.0       
        total_token_count = 0  
        start_time = time.time()

        for protein_feat, tgt_seq, _  in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            protein_feat = protein_feat.to(device)
            tgt_seq = tgt_seq.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(protein_feat, tgt_seq[:, :-1])
            labels = tgt_seq[:, 1:]

            loss_sum = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

            valid_tokens = (labels != pad_id).sum().item()
            loss =loss_sum/valid_tokens

            loss.backward()
            optimizer.step()

            total_loss_sum     += loss_sum.item()
            total_token_count  += valid_tokens

        avg_loss = total_loss_sum / max(1, total_token_count)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s", flush=True)

    return loss_history
