from pathlib import Path

import pandas as pd
import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np


def json_to_df(dataset_root_dir, train=True):
    json_file_location = Path(dataset_root_dir) / f"{'train' if train else 'test'}.jsonl"
    chunks = pd.read_json(json_file_location, lines=True, chunksize=100_000)
    df = pd.DataFrame()

    for e, chunk in tqdm.tqdm(enumerate(chunks), "Loading data", chunks.nrows):
        event_dict = {
            'session': [],
            'aid': [],
            'ts': [],
            'type': [],
        }
        if e < 2:
            # train_sessions = pd.concat([train_sessions, chunk])
            for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
                for event in events:
                    event_dict['session'].append(session)
                    event_dict['aid'].append(event['aid'])
                    event_dict['ts'].append(event['ts'])
                    event_dict['type'].append(event['type'])
            chunk_session = pd.DataFrame(event_dict)
            df = pd.concat([df, chunk_session])
        else:
            break

    df = df.reset_index(drop=True)
    return df


def per_session_action_sequences(df):
    d = {}
    for index, (session, aid, ts, type) in df.sort_values(by=['ts']).iterrows():
        if session in d:
            d[session].append(type)
        else:
            d[session] = [type]
    return d


class ActionSequenceDataset(Dataset):
    PAD_token = 5
    SOS_token = 3
    EOS_token = 4

    action_to_token = {
        'clicks': 0,
        'carts': 1,
        'orders': 2
    }

    def __init__(self, dataset_root_dir, min_session_length=15, max_session_length=20, train=True):
        # Settings
        self.sequence_length = 10

        # Load data
        df = json_to_df(dataset_root_dir, train)

        # Select the session we think are interesting
        vc = df.session.value_counts()
        selected_session_ids = vc[vc.between(min_session_length, max_session_length)].index
        selected_sessions = df.loc[df.session.isin(selected_session_ids)]

        # Create an array of action sequences
        self.action_sequences = per_session_action_sequences(selected_sessions).values()
        self.action_sequences = [[self.action_to_token[action] for action in seq] for seq in self.action_sequences]
        self.action_sequences = [np.asarray(seq, dtype=np.int64) for seq in self.action_sequences]

        print(f"Loaded {len(self.action_sequences)} action sequences.")

    def __len__(self):
        return len(self.action_sequences)

    def __getitem__(self, idx):
        action_sequence = self.action_sequences[idx]

        # Split the sequence into input and target
        X = action_sequence[:self.sequence_length]
        y = action_sequence[self.sequence_length:]

        # Add the start and end tokens
        X = np.concatenate(([self.SOS_token], X, [self.EOS_token]))
        y = np.concatenate(([self.SOS_token], y, [self.EOS_token]))

        # Pad the target
        if len(y) < (self.sequence_length + 2):
            y = np.concatenate((y, [self.PAD_token] * ((self.sequence_length + 2) - len(y))))

        return X, y


def to_stupid_batches(dataloader):
    batches = []
    for batched_X, batched_y in dataloader:
        batch = []
        for X, y in zip(batched_X.numpy(), batched_y.numpy()):
            batch.append([X, y])
        batches.append(np.asarray(batch, dtype=np.int64))
    return batches


def load_dataset_as_stupid_batches(dataset_root_dir, train):
    ds = ActionSequenceDataset(dataset_root_dir, train=train)
    dataloader = DataLoader(ds, batch_size=16, shuffle=train)
    return to_stupid_batches(dataloader)


if __name__ == '__main__':
    dataset_root_dir = r"C:\Projects\datasets\otto-recommender-system"
    train_dataloader = load_dataset_as_stupid_batches(dataset_root_dir, train=True)
    test_dataloader = load_dataset_as_stupid_batches(dataset_root_dir, train=False)

    for batch in train_dataloader:
        for x, y in batch:
            print(f"{x} -> {y}")

