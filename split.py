import numpy as np

from tools import generate_scaffold


def stratified_scaffold_split(data, smiles_column, val_size=0.1, test_size=0.1, seed=9527):
    np.random.seed(seed)
    scaffolds, no_scaffold_indices = {}, []

    for i, smiles in enumerate(data[smiles_column]):
        scaffold = generate_scaffold(smiles)
        if scaffold:
            scaffolds.setdefault(scaffold, []).append(i)
        else:
            no_scaffold_indices.append(i)

    total_size = len(data)
    target_val_size = int(total_size * val_size)
    target_test_size = int(total_size * test_size)
    target_train_size = total_size - target_val_size - target_test_size
    total_target = target_train_size + target_val_size + target_test_size
    train_ratio = target_train_size / total_target
    val_ratio = target_val_size / total_target
    test_ratio = target_test_size / total_target

    train_indices, val_indices, test_indices = [], [], []

    for scaffold in sorted(scaffolds.keys(), key=lambda s: len(scaffolds[s]), reverse=True):
        indices = scaffolds[scaffold].copy()
        np.random.shuffle(indices)
        remaining = len(indices)
        if remaining == 0:
            continue

        if remaining >= 3:
            alloc_train = round(remaining * train_ratio)
            alloc_val = round(remaining * val_ratio)
            alloc_test = remaining - alloc_train - alloc_val
            if alloc_train + alloc_val + alloc_test != remaining:
                diff = remaining - (alloc_train + alloc_val + alloc_test)
                alloc_train += diff
        elif remaining == 3:
            alloc_train, alloc_val, alloc_test = 1, 1, 1
        elif remaining == 2:
            alloc_train, alloc_val, alloc_test = 1, 1, 0
        else:
            alloc_train, alloc_val, alloc_test = 1, 0, 0

        alloc_train = min(alloc_train, remaining)
        remaining_after_train = remaining - alloc_train
        alloc_val = min(alloc_val, remaining_after_train)
        alloc_test = remaining_after_train - alloc_val

        train_indices.extend(indices[:alloc_train])
        val_indices.extend(indices[alloc_train:alloc_train + alloc_val])
        test_indices.extend(indices[alloc_train + alloc_val:alloc_train + alloc_val + alloc_test])

    if no_scaffold_indices:
        indices = no_scaffold_indices.copy()
        np.random.shuffle(indices)
        remaining = len(indices)
        if remaining > 0:
            if remaining >= 3:
                alloc_train = round(remaining * train_ratio)
                alloc_val = round(remaining * val_ratio)
                alloc_test = remaining - alloc_train - alloc_val
                if alloc_train + alloc_val + alloc_test != remaining:
                    diff = remaining - (alloc_train + alloc_val + alloc_test)
                    alloc_train += diff
            elif remaining == 3:
                alloc_train, alloc_val, alloc_test = 1, 1, 1
            elif remaining == 2:
                alloc_train, alloc_val, alloc_test = 1, 1, 0
            else:
                alloc_train, alloc_val, alloc_test = 1, 0, 0

            alloc_train = min(alloc_train, remaining)
            remaining_after_train = remaining - alloc_train
            alloc_val = min(alloc_val, remaining_after_train)
            alloc_test = remaining_after_train - alloc_val

            train_indices.extend(indices[:alloc_train])
            val_indices.extend(indices[alloc_train:alloc_train + alloc_val])
            test_indices.extend(indices[alloc_train + alloc_val:alloc_train + alloc_val + alloc_test])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def split_data(data, smiles_column="SMILES", target_column="log IC50", val_size=0.1, test_size=0.1, random_state=9527):
    apol_col_index = data.columns.get_loc("apol")
    feature_indices = list(range(apol_col_index, len(data.columns)))

    train_indices, val_indices, test_indices = stratified_scaffold_split(
        data, smiles_column, val_size=val_size, test_size=test_size, seed=random_state
    )

    train_data = data.iloc[train_indices].reset_index(drop=True)
    val_data = data.iloc[val_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)

    X_train = train_data.iloc[:, feature_indices].values
    X_val = val_data.iloc[:, feature_indices].values
    X_test = test_data.iloc[:, feature_indices].values
    y_train = train_data[target_column].values
    y_val = val_data[target_column].values
    y_test = test_data[target_column].values

    return X_train, X_val, X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices


def save_splits(data, train_idx, val_idx, test_idx, split_dir):
    data.iloc[train_idx].to_csv(f"./{split_dir}/train_data.csv", index=False)
    data.iloc[val_idx].to_csv(f"./{split_dir}/val_data.csv", index=False)
    data.iloc[test_idx].to_csv(f"./{split_dir}/test_data.csv", index=False)