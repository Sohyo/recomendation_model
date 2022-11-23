import torch
import torch.nn as nn

from data.dataset import load_dataset_as_stupid_batches
from test_model import Transformer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=6, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
# model = torch.load('saved_model.pth')


def train_loop(model, opt, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, 'Train loop', len(dataloader)):
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list


def predict(model, input_sequence, max_length=15, SOS_token=3, EOS_token=4):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


def inference_with_val_data(dataloader):
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

        for single_x, single_y in zip(X, y):
            result = predict(model, single_x.unsqueeze(0))
            print(f"Input: {single_x.view(-1).tolist()[1:-1]}")
            print(f"Continuation: {result[1:-1]}")
            print(f"Expected continuation: {single_y[1:-1]}")
        exit()


def main():
    # from old_data import train_dataloader, val_dataloader

    dataset_root_dir = r"C:\Projects\datasets\otto-recommender-system"
    train_dataloader = load_dataset_as_stupid_batches(dataset_root_dir, train=True)
    val_dataloader = load_dataset_as_stupid_batches(dataset_root_dir, train=False)

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 5)
    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 0, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 1, 1, 0, 1, 1, 0, 1, 1, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 1, 1, 1, 1, 1, 1, 1, 0, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 0, 0, 0, 0, 1, 0, 1, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 4]], dtype=torch.long, device=device),
        torch.tensor([[3, 0, 1, 4]], dtype=torch.long, device=device)
    ]

    model.eval()
    for idx, example in enumerate(examples):
        result = predict(model, example)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()

    torch.save(model, 'saved_model.pth')


if __name__ == "__main__":
    main()
