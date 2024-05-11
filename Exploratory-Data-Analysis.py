'''Q1) Plot the class distribution for the chosen dataset(s). If you are working with different datasets, compare the distributions. Attach your plots and code snippets.'''

'''First, we need to download the reddit_tifu dataset (since the dataset is in the croissant format, it is available to download in the Hugging Face model hub). If we don't have the datasets library installed, install it using the following command in your terminal.'''
!pip install datasets

'''Then download the dataset as follows:'''
from datasets import load_dataset
reddit_tifu = load_dataset('reddit_tifu', 'long', split='train')

'''Now let's create a PyTorch DataLoader to iterate over the dataset:'''
import torch
from torch.utils.data import Dataset, DataLoader

class RedditTIFUDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        ''' Converting the text data to lists '''
        docs_list = data['documents']
        title_list = data['title']
        tldr_list = data['tldr']

        ''' Copying the 'ups', 'num_comments', 'score', 'upvote_ratio' to a new dictionary '''
        additional_data = {
            'ups': data['ups'],
            'num_comments': data['num_comments'],
            'score': data['score'],
            'upvote_ratio': data['upvote_ratio']
        }

        return docs_list, title_list, tldr_list, additional_data

''' Prepare the dataset for the DataLoader '''
reddit_tifu_dataset = RedditTIFUDataset(reddit_tifu)

''' Define the batch size and create a PyTorch DataLoader '''
batch_size = 32
reddit_tifu_loader = DataLoader(reddit_tifu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

''' Answer 1) In order to plot the class-distribution for the reddit_tifu dataset, we must use Pandas DataFrame: 
1.  Store the data with an appropriate dataset handler.
2. Import the required libraries: matplotlib.pyplot and wordcloud.
3. Combine all documents in the reddit_tifu_df DataFrame into a single string using the join() method.
4. Create a WordCloud object with a width of 800 pixels, height of 400 pixels, and a random state of 42. Generate the word cloud using the generate() method and pass the combined string as an argument.
5. Create a new figure with a size of 10x5 inches using plt.figure().
Display the word cloud using plt.imshow() and set the interpolation to 'bilinear'. 

Storing the dataset in a Pandas DataFrame: '''
import pandas as pd

''' Convert the DataLoader to a Pandas DataFrame '''
def data_loader_to_data_frame(data_loader):
    all_data = []
    for batch in data_loader:
        docs_list, title_list, tldr_list, additional_data = batch
        for doc, title, tldr in zip(docs_list, title_list, tldr_list):
            data = {
                'documents': doc,
                'title': title,
                'tldr': tldr,
                **additional_data
            }
            all_data.append(data)

    df = pd.DataFrame(all_data)
    return df

reddit_tifu_df = data_loader_to_data_frame(reddit_tifu_loader)

''' Since the reddit_tifu dataset is a text dataset without clear categories, we can't create a class distribution plot. However, we can create a word cloud to visualize the most frequently used words in any of the fields. In this case, we have chosen the 'documents' column.

First, let's install the wordcloud library: '''
!pip install wordcloud

''' Now, let's create a word cloud for the 'documents' column: '''
import matplotlib.pyplot as plt
from wordcloud import WordCloud

''' Combine all documents into a single string '''
all_documents = " ".join(reddit_tifu_df['documents'])

''' Create a WordCloud object '''
wordcloud = WordCloud(width=800, height=400, random_state=42).generate(all_documents)

''' Plot the word cloud '''
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

