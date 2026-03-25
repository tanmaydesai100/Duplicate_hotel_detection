# import pandas as pd
# import yaml
# from sklearn.model_selection import train_test_split

# def load_config(path="../../configs/config.yaml"):
#     return yaml.safe_load(open(path))

# if __name__=="__main__":
#     cfg = load_config()
#     # Read original dataset
#     df = pd.read_csv("../data/final_dataset.csv")
#     # Split into train/test
#     df_train, df_test = train_test_split(
#         df, test_size=cfg['data']['test_size'],
#         random_state=cfg['data']['random_seed'],
#         stratify=df['Match'], shuffle=True
#     )
#     # Write training pairs
#     train_df = df_train.rename(
#         columns={'addressA':'text_a','addressB':'text_b','Match':'label'}
#     )[['text_a','text_b','label']]
#     train_df.to_csv("../data/hotel_pairs.csv", index=False)

#     # Build inference_input from test split
#     unique_a = df_test[['orig_indexA','addressA','lat_indexA','lon_indexA']].rename(
#         columns={'orig_indexA':'orig_index','addressA':'text','lat_indexA':'lat','lon_indexA':'long'}
#     )
#     unique_b = df_test[['orig_indexB','addressB','lat_indexB','lon_indexB']].rename(
#         columns={'orig_indexB':'orig_index','addressB':'text','lat_indexB':'lat','lon_indexB':'long'}
#     )
#     inf_df = pd.concat([unique_a, unique_b], ignore_index=True)
#     inf_df = inf_df.drop_duplicates(subset=['text']).reset_index(drop=True)
#     inf_df.to_csv("../data/inference_input.csv", index=False)
#new commented one
# import pandas as pd
# import yaml
# from sklearn.model_selection import train_test_split

# def load_config(path="../../configs/config.yaml"):
#     return yaml.safe_load(open(path))

# if __name__ == "__main__":
#     cfg = load_config()
#     # Read original dataset
#     df = pd.read_csv("../data/final_dataset.csv")

#     # Split into train/test
#     df_train, df_test = train_test_split(
#         df,
#         test_size=cfg['data']['test_size'],
#         random_state=cfg['data']['random_seed'],
#         stratify=df['Match'],
#         shuffle=True
#     )

#     # Write training pairs
#     train_df = df_train.rename(
#         columns={'addressA': 'text_a', 'addressB': 'text_b', 'Match': 'label'}
#     )[['text_a', 'text_b', 'label']]
#     train_df.to_csv("../data/hotel_pairs.csv", index=False)

#     # Build inference_input from test split
#     # We need unique hotels with their match group for later accuracy computation
#     # Prepare A-side
#     unique_a = df_test[['orig_indexA', 'addressA', 'lat_indexA', 'lon_indexA', 'Match']].rename(
#         columns={
#             'orig_indexA': 'orig_index',
#             'addressA': 'text',
#             'lat_indexA': 'lat',
#             'lon_indexA': 'long',
#             'Match': 'match_id'
#         }
#     )
#     # Prepare B-side
#     unique_b = df_test[['orig_indexB', 'addressB', 'lat_indexB', 'lon_indexB', 'Match']].rename(
#         columns={
#             'orig_indexB': 'orig_index',
#             'addressB': 'text',
#             'lat_indexB': 'lat',
#             'lon_indexB': 'long',
#             'Match': 'match_id'
#         }
#     )
#     # Combine and dedupe on text
#     inf_df = pd.concat([unique_a, unique_b], ignore_index=True)
#     inf_df = inf_df.drop_duplicates(subset=['orig_index']).reset_index(drop=True)

#     # Write inference inputs, including match_id for later ground-truth comparison
#     inf_df.to_csv("../data/inference_input.csv", index=False)
#     print("Saved inference_input.csv with columns: ", inf_df.columns.tolist())

#     # Also save the test pairs with true labels for direct pairwise comparison
#     # This file can be used to compute accuracy by comparing to model output
#     test_pairs = df_test[['orig_indexA', 'orig_indexB', 'Match']].rename(
#         columns={'orig_indexA': 'orig_indexA', 'orig_indexB': 'orig_indexB', 'Match': 'is_duplicate'}
#     )
#     test_pairs.to_csv("../data/inference_labels.csv", index=False)
#     print("Saved inference_labels.csv with pairwise true labels: ", test_pairs.columns.tolist())
# import pandas as pd
# import yaml
# from sklearn.model_selection import train_test_split

# def load_config(path="../configs/config.yaml"):
#     return yaml.safe_load(open(path))

# if __name__ == "__main__":
#     cfg = load_config()
#     # Read original dataset
#     df = pd.read_csv("../data/final_dataset.csv")

#     # Split into train/test
#     df_train, df_test = train_test_split(
#         df,
#         test_size=cfg['data']['test_size'],
#         random_state=cfg['data']['random_seed'],
#         stratify=df['Match'],
#         shuffle=True
#     )

#     # Write training pairs
#     train_df = df_train.rename(
#         columns={'addressA': 'text_a', 'addressB': 'text_b', 'Match': 'label'}
#     )[['text_a', 'text_b', 'label']]
#     train_df.to_csv("../data/hotel_pairs.csv", index=False)

#     # Build inference_input from test split
#     # We need unique hotels with their match group for later accuracy computation
#     # Prepare A-side
#     unique_a = df_test[['orig_indexA', 'addressA', 'lat_indexA', 'lon_indexA', 'Match']].rename(
#         columns={
#             'orig_indexA': 'orig_index',
#             'addressA': 'text',
#             'lat_indexA': 'lat',
#             'lon_indexA': 'long',
#             'Match': 'match_id'
#         }
#     )
#     # Prepare B-side
#     unique_b = df_test[['orig_indexB', 'addressB', 'lat_indexB', 'lon_indexB', 'Match']].rename(
#         columns={
#             'orig_indexB': 'orig_index',
#             'addressB': 'text',
#             'lat_indexB': 'lat',
#             'lon_indexB': 'long',
#             'Match': 'match_id'
#         }
#     )
#     # Combine and dedupe on text
#     inf_df = pd.concat([unique_a, unique_b], ignore_index=True)
#     inf_df = inf_df.drop_duplicates(subset=['orig_index']).reset_index(drop=True)

#     # Write inference inputs, including match_id for later ground-truth comparison
#     inf_df.to_csv("../data/inference_input.csv", index=False)
#     print("Saved inference_input.csv with columns: ", inf_df.columns.tolist())

#     # Also save the test pairs with true labels for direct pairwise comparison
#     # This file can be used to compute accuracy by comparing to model output
#     test_pairs = df_test[['orig_indexA', 'orig_indexB', 'Match']].rename(
#         columns={'orig_indexA': 'orig_indexA', 'orig_indexB': 'orig_indexB', 'Match': 'is_duplicate'}
#     )
#     test_pairs.to_csv("../data/inference_labels.csv", index=False)
#     print("Saved inference_labels.csv with pairwise true labels: ", test_pairs.columns.tolist())
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(path="../configs/config.yaml"):
    return yaml.safe_load(open(path))

if __name__ == "__main__":
    cfg = load_config()
    # Read original dataset
    df = pd.read_csv("../data/final_dataset.csv")

    # Split into train/test
    df_train, df_test = train_test_split(
        df,
        test_size=cfg['data']['test_size'],
        random_state=cfg['data']['random_seed'],
        stratify=df['Match'],
        shuffle=True
    )

    # Write training pairs (unchanged)
    train_df = df_train.rename(
        columns={'addressA': 'text_a', 'addressB': 'text_b', 'Match': 'label'}
    )[['text_a', 'text_b', 'label']]
    train_df.to_csv("../data/hotel_pairs.csv", index=False)

    # Build inference_input from test split
    # Only need unique hotel records with coords and text
    unique_a = df_test[['orig_indexA', 'addressA', 'lat_indexA', 'lon_indexA']].rename(
        columns={
            'orig_indexA': 'orig_index',
            'addressA': 'text',
            'lat_indexA': 'lat',
            'lon_indexA': 'long'
        }
    )
    unique_b = df_test[['orig_indexB', 'addressB', 'lat_indexB', 'lon_indexB']].rename(
        columns={
            'orig_indexB': 'orig_index',
            'addressB': 'text',
            'lat_indexB': 'lat',
            'lon_indexB': 'long'
        }
    )
    inf_df = pd.concat([unique_a, unique_b], ignore_index=True)
    inf_df = inf_df.drop_duplicates(subset=['orig_index']).reset_index(drop=True)

    # Write inference inputs without match_id
    inf_df.to_csv("../data/inference_input.csv", index=False)
    print("Saved inference_input.csv with columns: ", inf_df.columns.tolist())

    # Save only true duplicate pairs for evaluation
    dup_pairs = df_test[df_test['Match'] == 1][['orig_indexA', 'orig_indexB']].rename(
        columns={'orig_indexA': 'orig_indexA', 'orig_indexB': 'orig_indexB'}
    )
    dup_pairs.to_csv("../data/inference_labels.csv", index=False)
    print("Saved inference_labels.csv with duplicate-only pairs: ", dup_pairs.shape[0], "rows")

