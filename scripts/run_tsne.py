from argparse import ArgumentParser

import pandas as pd
from sklearn.manifold import TSNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--emb_csv",
        required=True,
        type=str,
        help="path to embedding csv",
    )
    parser.add_argument(
        "--tsne_csv",
        required=True,
        type=str,
        help="path to save tsne csv to",
    )
    args, unk_args = parser.parse_known_args()

    aves_frame_df = pd.read_csv(args.emb_csv, index_col='emb_idx')

    emb_cols = [col for col in aves_frame_df.columns if col.startswith('emb')]
    aves_frame_embs = aves_frame_df[emb_cols].values

    aves_tsne = TSNE(n_components=3, learning_rate=100, perplexity=30, verbose=2, angle=0.1, n_jobs=-1).fit_transform(
        aves_frame_embs
    )

    tsne_dict = {f'tsne_component_{i}': aves_tsne[:, i] for i in range(aves_tsne.shape[-1])}

    # gather tsne components & ids in a dataframe
    aves_tsne_df = pd.DataFrame.from_dict(
        {'emb_idx': aves_frame_df.index, 'voc_idx': aves_frame_df['voc_idx']} | tsne_dict
    ).set_index('emb_idx')

    aves_tsne_df.to_csv(args.tsne_csv)
