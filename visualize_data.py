import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import get_splits, SplitConfig

def create_visualizations(ds):
    df = pd.DataFrame(ds["train"])
    
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot 1: Distribution of Review Lengths by Sentiment
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='word_count', hue='label', element="step", kde=True)
    plt.title('Distribution of Review Word Counts')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend(title='Sentiment', labels=['Positive', 'Negative'])

    # Plot 2: Class Balance
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='label')
    plt.title('Class Balance (Training Set)')
    plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig("results/data_exploration.png")
    print("Visualization saved to results/data_exploration.png")
    plt.show()

if __name__ == "__main__":
    cfg = SplitConfig()
    ds = get_splits(cfg=cfg)
    create_visualizations(ds)