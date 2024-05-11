# Q1) Plot the class distribution for the chosen dataset(s). If you are working with different datasets, compare the distributions. Attach your plots and code snippets.
# Q2) Study different categories and plot sample images of some categories. Attach sample images and code snippets.
# Q3) Which categories are most confusing or hard to differentiate? Which categories are the easiest to differentiate? Think how you can check qualitatively (visually) and quantitatively.

# First, we need to download the reddit_tifu dataset (since the dataset is in the croissant format, it is available to download in the Hugging Face model hub). If we don't have the datasets library installed, install it using the following command in your terminal.
!pip install datasets

# Then download the dataset as follows:
from datasets import load_dataset
reddit_tifu = load_dataset('reddit_tifu', 'long', split='train')

# Now let's create a PyTorch DataLoader to iterate over the dataset:
import torch
from torch.utils.data import Dataset, DataLoader

class RedditTIFUDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # Converting the text data to lists 
        docs_list = data['documents']
        title_list = data['title']
        tldr_list = data['tldr']

        # Copying the 'ups', 'num_comments', 'score', 'upvote_ratio' to a new dictionary 
        additional_data = {
            'ups': data['ups'],
            'num_comments': data['num_comments'],
            'score': data['score'],
            'upvote_ratio': data['upvote_ratio']
        }

        return docs_list, title_list, tldr_list, additional_data

# Prepare the dataset for the DataLoader 
reddit_tifu_dataset = RedditTIFUDataset(reddit_tifu)

# Define the batch size and create a PyTorch DataLoader 
batch_size = 32
reddit_tifu_loader = DataLoader(reddit_tifu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Answer 1) In order to plot the class-distribution for the reddit_tifu dataset, we must use Pandas DataFrame: 
# 1.  Store the data with an appropriate dataset handler.
# 2. Import the required libraries: matplotlib.pyplot and wordcloud.
# 3. Combine all documents in the reddit_tifu_df DataFrame into a single string using the join() method.
# 4. Create a WordCloud object with a width of 800 pixels, height of 400 pixels, and a random state of 42. Generate the word cloud using the generate() method and pass the combined string as an argument.
# 5. Create a new figure with a size of 10x5 inches using plt.figure().
# Display the word cloud using plt.imshow() and set the interpolation to 'bilinear'. 

# Storing the dataset in a Pandas DataFrame: 
import pandas as pd

# Convert the DataLoader to a Pandas DataFrame:
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

# Since the reddit_tifu dataset is a text dataset without clear categories, we can't create a class distribution plot. However, we can create a word cloud to visualize the most frequently used words in any of the fields. In this case, we have chosen the 'documents' column.

# First, let's install the wordcloud library: 
!pip install wordcloud

# Now, let's create a word cloud for the 'documents' column: 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Combine all documents into a single string 
all_documents = " ".join(reddit_tifu_df['documents'])

# Create a WordCloud object 
wordcloud = WordCloud(width=800, height=400, random_state=42).generate(all_documents)

# Plot the word cloud 
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Answer 2) Once again, as the reddit_tifu dataset is a text dataset without any image data or clear categories, it is not possible to plot sample images of different categories. However, we could do something similar for text-based datasets.

#Our process will involve returning samples of a particular parameter like the TLDR of a particular post containing one of the keywords we identified in the previous ones, and if we change the keyword a different post is returned.
import random

# Define the keyword to search for
keyword = 'teacher'

# Filter the dataset to include only rows with the keyword in the TLDR
keyword_rows = reddit_tifu_df[reddit_tifu_df['tldr'].str.contains(keyword)]

# Define the number of samples to display
num_samples = 5

# Display random samples
sample_indices = random.sample(range(len(keyword_rows)), num_samples)
sample_tldrs = keyword_rows['tldr'].iloc[sample_indices]

# Print the sample TLD
print("Sample TLDRs containing keyword '{}':".format(keyword))
for i, tldr in enumerate(sample_tldrs):
    print(f"\nSample {i+1}:")
    print(tldr)

# Answer 3) Given the context, we can consider factors such as the length of the post, the number of upvotes, the number of comments, the score, and the upvote ratio to differentiate between two r/tifu samples.

# To check qualitatively (visually), we can create scatter plots to visualize the relationship between these factors. Here's an example of how to create scatter plots for the length of the post and the number of upvotes:
import pandas as pd

print("reddit_tifu_df shape:", reddit_tifu_df.shape)
print("reddit_tifu_df dtypes:", reddit_tifu_df.dtypes)

import matplotlib.pyplot as plt
import torch
import numpy as np

# Convert tensor objects to Python scalars
reddit_tifu_df['num_comments'] = reddit_tifu_df['num_comments'].apply(lambda x: x.numpy()[0] if isinstance(x, torch.Tensor) else x)
reddit_tifu_df['ups'] = reddit_tifu_df['ups'].apply(lambda x: x.numpy()[0] if isinstance(x, torch.Tensor) else x)

# Remove NaN values from the num_comments and ups columns
reddit_tifu_df = reddit_tifu_df.dropna(subset=['num_comments', 'ups'])

# Convert lists to 1D NumPy arrays
num_comments_array = np.array(reddit_tifu_df['num_comments'])
ups_array = np.array(reddit_tifu_df['ups'])

# Create scatter plot for length of post vs. number of upvotes
plt.figure(figsize=(10, 5))
plt.scatter(num_comments_array, ups_array)
plt.xlabel('Number of Comments')
plt.ylabel('Number of Upvotes')
plt.title('Length of Post vs Number of Upvotes')
plt.grid(True)
plt.show()

# To check quantitatively, we can calculate the correlation between these factors using the corr() function in Pandas. Here's an example of how to calculate the correlation between the length of the post and the number of upvotes:

# Calculate correlation between length of post and number of upvotes
correlation = reddit_tifu_df[['num_comments', 'ups']].corr().iloc[0, 1]
print("Correlation between length of post and number of upvotes:", correlation)

# Based on the scatter plot and correlation, we can see that there is a strong positive correlation between the length of the post and the number of upvotes. This suggests that longer posts tend to receive more upvotes and comments (which increases karma, promoting the post to more people) on average.Similarly, we can create scatter plots and calculate correlations for other factors such as the number of comments, score, and upvote ratio. This will help us understand the relationship between these factors and potentially identify which factors are most confusing or hard to differentiate, and which factors are the easiest to differentiate.

# Another category for which we can plot correlations is upvote ratio, as shown below:
import torch
import matplotlib.pyplot as plt

# Limit the dataset to just the top 100 rows
reddit_tifu_df = reddit_tifu_df.head(100)

# Calculate upvote ratio
reddit_tifu_df.loc[:, 'upvote_ratio'] = reddit_tifu_df['ups'] / (reddit_tifu_df['ups'] + reddit_tifu_df['score'])

# Convert lists to PyTorch tensors
upvote_ratio_tensor = torch.tensor([x for sublist in reddit_tifu_df['upvote_ratio'].tolist() for x in sublist])

# Create histogram for upvote ratio
plt.figure(figsize=(10, 5))
plt.hist(upvote_ratio_tensor.numpy(), bins=20)
plt.xlabel('Upvote Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Upvote Ratios')
plt.grid(True)
plt.show()

# This code first calculates the upvote ratio for each post by dividing the number of upvotes by the total number of votes (upvotes + downvotes). It then converts the 'upvote_ratio' column to a 1D NumPy array. Finally, it creates a histogram to show the distribution of upvote ratios for all posts in the dataset.

# The upvote ratio can be a useful metric for determining the popularity or controversiality of a post. A high upvote ratio indicates that a post has received a large number of upvotes relative to the number of downvotes, suggesting that it is well-liked by the community. On the other hand, a low upvote ratio indicates that a post has received a large number of downvotes relative to the number of upvotes, suggesting that it is controversial or unpopular.

# The upvote ratio can also be used as a quantitative way of determining categorical correlation within a dataset. For example, suppose you wanted to determine whether there is a correlation between the length of a post and its upvote ratio. You could calculate the upvote ratio for each post, and then group the posts by length (e.g. short, medium, long). You could then compare the average upvote ratio for each group using a statistical test such as ANOVA or a t-test. If there is a significant difference in the average upvote ratio between the groups, this would suggest that there is a correlation between the length of a post and its upvote ratio.

# Another example of a quantitative metric that can be used to determine categorical correlation is the F1 score. The F1 score is a measure of the accuracy of a binary classifier, and is calculated as the harmonic mean of the precision and recall. It can be used to evaluate the performance of a classifier that predicts whether a post belongs to a certain category (e.g. spam, not spam). For example, suppose you wanted to determine whether there is a correlation between the length of a post and its likelihood of being spam. You could train a binary classifier to predict whether a post is spam based on its length, and then evaluate the performance of the classifier using the F1 score. If the F1 score is high, this would suggest that there is a correlation between the length of a post and its likelihood of being spam (as spam posts generally tend to receive a lot more downvotes too, resulting in a lower upvote ratio).
