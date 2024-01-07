import torch
from torch.utils.data import DataLoader

from model import TransformerWithPE
from utils import (load_and_partition_data, make_datasets, move_to_device,
                   split_sequence, visualize)

BS = 512
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = 100
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001


def main() -> None:
    # Load data and generate train and test datasets / dataloaders
    sequences, num_features = load_and_partition_data("data.npz")
    train_set, test_set = make_datasets(sequences)
    train_loader, test_loader = DataLoader(
        train_set, batch_size=BS, shuffle=True
    ), DataLoader(test_set, batch_size=BS, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer and loss criterion
    model = TransformerWithPE(
        num_features, num_features, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # Train loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)
            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: "
            f"{(epoch_loss / len(train_loader)):.4f}"
        )

    # Evaluate model
    model.eval()
    eval_loss = 0.0
    infer_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)

            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            eval_loss += loss.item()

            # Run inference with model
            pred_infer = model.infer(src, tgt.shape[1])
            loss_infer = criterion(pred_infer, tgt_y)
            infer_loss += loss_infer.item()

            if idx < NUM_VIS_EXAMPLES:
                visualize(src, tgt, pred, pred_infer)

    avg_eval_loss = eval_loss / len(test_loader)
    avg_infer_loss = infer_loss / len(test_loader)

    print(f"Eval / Infer Loss on test set: {avg_eval_loss:.4f} / {avg_infer_loss:.4f}")


if __name__ == "__main__":
    main()
