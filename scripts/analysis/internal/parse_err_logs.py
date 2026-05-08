"""Parse Chemprop verbose logs and export them to TensorBoard event files.

This utility was used to inspect ensemble-member training curves in TensorBoard
without modifying the original Chemprop training loop.
"""

import re
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def parse_verbose_log(file_path):
    """Parse one Chemprop `verbose.log` into train, validation, and test tables."""
    train_data, val_data, test_data = [], [], []
    
    current_model = -1
    iteration_counter = 0

    model_start_pattern = re.compile(r"Building model (\d+)")
    loss_pattern = re.compile(r"Loss = ([\d.e+-]+)")
    val_pattern = re.compile(r"Validation rmse = ([\d.e+-]+)")
    test_pattern = re.compile(r"Model (\d+) test rmse = ([\d.e+-]+)")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None, None

    with open(file_path, 'r') as f:
        for line in f:
            m_match = model_start_pattern.search(line)
            if m_match:
                current_model = int(m_match.group(1))
                iteration_counter = 0
                continue

            l_match = loss_pattern.search(line)
            if l_match and current_model != -1:
                train_data.append({
                    'model': current_model,
                    'step': iteration_counter,
                    'loss': float(l_match.group(1))
                })
                iteration_counter += 1

            v_match = val_pattern.search(line)
            if v_match and current_model != -1:
                val_data.append({
                    'model': current_model,
                    'step': iteration_counter,
                    'rmse': float(v_match.group(1))
                })

            t_match = test_pattern.search(line)
            if t_match:
                test_data.append({
                    'model': int(t_match.group(1)),
                    'test_rmse': float(t_match.group(2))
                })

    return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

def export_to_tensorboard(train_df, val_df, test_df, base_log_dir="runs/chemprop_verbose"):
    """Write one TensorBoard run per ensemble member for easy overlays."""
    if train_df.empty:
        print("No data found to export.")
        return

    models = train_df['model'].unique()
    print(f"Found data for models: {models}")

    for m_id in models:
        log_path = os.path.join(base_log_dir, f"model_{m_id}")
        writer = SummaryWriter(log_dir=log_path)

        m_train = train_df[train_df['model'] == m_id]
        for _, row in m_train.iterrows():
            writer.add_scalar("Loss/Train", row['loss'], row['step'])

        m_val = val_df[val_df['model'] == m_id]
        for _, row in m_val.iterrows():
            writer.add_scalar("Metric/Val_RMSE", row['rmse'], row['step'])

        m_test = test_df[test_df['model'] == m_id]
        if not m_test.empty:
            final_step = m_train['step'].max()
            writer.add_scalar("Metric/Test_RMSE", m_test.iloc[0]['test_rmse'], final_step)

        writer.close()
    
    print(f"Successfully exported {len(models)} models to {base_log_dir}")

TARGET = "abs"
DATASET = "dsscdb"
Ns = ["N01376_s42", "N01000_s42", "N00250_s42", "N00100_s42", "N00050_s42"]

for N in Ns:
    LOG_PATH = f"checkpoints/{TARGET}/{DATASET}/scaffold/morgan_fingerprint/fromscratch_{N}/verbose.log"
    train, val, test = parse_verbose_log(LOG_PATH)
    export_to_tensorboard(train, val, test, base_log_dir=f"runs/{TARGET}/{DATASET}/{N}/")
