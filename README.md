![NLP](https://user-images.githubusercontent.com/36130927/127267409-09691ad8-807f-470b-96a9-5083e5b32fac.jpg)

# News Classification with Natural Language Processing and Deep Learning

*News acts as a trigger to drive a stock price either up or down, depending on the type of news. Some news creates a more longer-term effect, which may push the stock price upward for a few days up to a few weeks. It is also true for the opposite. Some news can plummet the stock price to close to nothing, creating ripple effects in consecutive days of play. Whereas some news has little or no effect. So, instead of classifying the news as “positive”, “negative”, or “neutral” for sentiment analysis, which most people on the Internet do, I classify news using a different methodology – news types. For examples, an earnings beater for a company may most likely have a much more positive effect than the company giving out their positive guidance for their next fiscal year. Or, a price target decrease announced by couple street analysts may have a more negative effect than a failed product launch. The whole model usually would combine NLP with other statistical modeling to yield the best prediction, but NLP is the first step of finding the right catalyst.*

## 1. Data
I downloaded the financial news data from one of the Kaggle datasets. In fact, any news headline datasets will do the work after being categorized.

There are only two columns needed for this analysis – headline and type. At first, the type column contains only empty values because it is newly created. Our job is to categorize as much news in the headline as possible.

![1 raw news](https://user-images.githubusercontent.com/36130927/127267706-cdb17366-434f-43af-a4a9-ead965fbc2c1.png)

Subsequently, we will determine which categories are needed for classification. In this project, we classify news in 7 types.
1.	Redundant/meaningless. Any news that has little or no impact on stock price movement.
2.	Analyst action. Initial coverage, ratings or price target adjustment by analysts.
3.	Earnings. Earnings performance or related news.
4.	Corporate action. Dividend or share repurchase program announcement, new contract or partnership, settlement of litigation, etc.
5.	Merger and acquisition. M&A with other companies.
6.	Company guidance. Company providing estimates guidance to their future performance.
7.	Options. Any news related to options or derivatives activities of the company.


## 2. Method
In this project, we will compare two methods and ultimately select one of them as our go-to model for news classification going forward.

1. **Rules-based:** I classified the news using keywords in 7 categories
2. **Manually-labelled:** I manually labelled close to 10,000 headlines into the same 7 categories

## 3. Data Wrangling

In the rules-based model, 30% comes from Analyst Action, 26% from Earnings, 21% Corporate Action, 14% Redundant/Meaningless. And the minority includes Company Guidance, Merger and Acquisition, and Options. There is a total of 418,013 news headlines in the dataset. The range from 1% to 30% shows imbalance in data. But let’s proceed for now and see the results first.

![2 type value counts ai](https://user-images.githubusercontent.com/36130927/127267986-849b4319-833c-46e7-935d-8f2496e1add1.png)

In the manually-labelled model, 25% from Redundant/Meaningless, 17% from Analyst Action, 16% Merger and Acquisition, 14% Earnings, 13% Options, and the rest includes Corporate Action and Company Guidance. This is a more balanced dataset, which contains 9,916 news headlines. However, it is just 2.4% of the quantity compared to the rules-based model above.

![2a type value counts manual](https://user-images.githubusercontent.com/36130927/127267989-8be979ee-6073-499e-ab36-007419b6b4c9.png)


## 4. Data Preprocessing

**Tokenization:** we use tokenization to break down text strings into chunks of words, wherein each word is called a token. These tokens will then be passed on to some other form of processing for further analysis.

**Sequence padding:** we now convert words into sequences of integers that is indexed with the vocabulary words. Then, we pad sequences into all same-length sequences. The process is mandatory because all neural networks require inputs to have the same shape and size, as it is reading an array as an input.




## 5. Modeling

**Sequential model:** I used sequential model because it is a linear stack of layers that creates layer-by-layer for problems. It is straightforward and easy to set up compared to functional model, which is more flexible and capable of creating complex networks that specifically connect different input and output layers.


**Pooling layers:** pooling layers use pooling operation to calculate the value in each patch of each feature map. The problem with feature maps is that they record precise position of the input features, meaning small movements in the position of the input feature will result in a different feature map. By using pooling layers, the results are down sampled (or pooled) feature maps that summarize the presence of features in the patches of the feature map.

Below is the model summary:

![4 sequential model](https://user-images.githubusercontent.com/36130927/127268120-6984cf95-09df-41db-a94d-550dfbe3da6b.png)


**Early stopping:** I also added early stopping to reduce the chance of overfitting the model. And with 2 patience, meaning the training will iterate 2 more times after early stopping kicks in, the rules-based model stops at 12 iterations with an accuracy of 98.2%.

![7 loss vs epoch ai](https://user-images.githubusercontent.com/36130927/127268126-3b50bfe3-3fb5-4365-b54b-376b5449f844.png)

Whereas the manually-labelled model stops at 34 iterations with an accuracy of 95.8%.

![7a loss vs epoch manual](https://user-images.githubusercontent.com/36130927/127268129-56f4dac4-9d40-4155-bc11-54ddcf83fcfb.png)


## 6. Testing

The simple made-up phrases below are used to test the validity of the model. It will also spot the weakness of the model in case of wrong classification.

The rules-based model got 9 out of 9 correct.

![8 testing ai](https://user-images.githubusercontent.com/36130927/127268140-61d4147d-db52-4242-9b76-b4de5d2845bf.png)

The manually-labelled model got 8 out of 9 correct, which shows some inconsistency in the model.

![8a testing manual](https://user-images.githubusercontent.com/36130927/127268144-4ea331d4-afd3-4116-8280-bc4cc607591c.png)



## 7. Prediction

Below is the result from the rules-based model. Out of the 20 news headlines, it classified 9 in earnings and 11 in Redundant/Meaningless. It is more sensitive than I would like when compared to my way of classification.

![9 predicting ai](https://user-images.githubusercontent.com/36130927/127268157-f34baad3-fd6b-4c53-ba27-23874e698ebd.png)

Below is the result from the manually-labelled model. We can tell 2 characteristics right off the bat.
1.	It classifies a lot of headlines as Redundant/Meaningless
2.	There is inconsistency of the method, e.g., the one headline classified an Analyst Action

![9a predicting manual](https://user-images.githubusercontent.com/36130927/127268161-eed6b8c0-e15e-4a1f-a7a3-23a6d19783f9.png)

There are definitely pros and cons within the manually-labelled model.

Pros:
-	It is more conservative. By identifying the majority of the headlines as meaningless news, we only focus on the few ones that may cause actions in the stock price movement

Cons:
-	As shown above, it misclassified 1 of the 9 made-up headline as Redundant/Meaningless, and 1 of the 20 most recent news headlines of Tesla as Analyst Action where it should be Redundant/Meaningless. This shows inconsistency in the model and needs more sample in the training set to improve its prediction.

With all being said, keep in mind that the way of classification is subjective. Some people may prefer the rules-based model as it captures all the news headlines as Earnings using the keywords. However, if I only focus on the absolutely essential news without other unnecessary distraction, the manually-labelled model fits my needs better.


## 8. Takeaways

This project tells us few key takeaways:
-	Early stopping is useful as it reduces the chance of overfitting the model.
-	Rules-based model produces a higher accuracy compared to manually-labelled model (98.2% vs 95.8%), but it does not mean the result is more favorable.
-	Although the manually-labelled model provides a more suitable result to my needs, it shows inconsistency in the model. It implies that an approximation of 1,000 samples in each category in the training set is not enough, and more samples are needed to improve its predictive power.


## 9. Further Work

Approaches can be considered for further improvement of the rules-based model:
-	Make the dataset more balanced by adding more news in the Options, Merger and Acquisition, and Company Guidance categories.
-	Try other neural network models and select other layers and add more neuron layers to further optimize the model.
-	Redefine rules with more or less keywords for classification.

Approaches can be considered for further improvement of the manually-labelled model:
-	Add more examples to the training set to improve consistency and prediction.
-	Similar to the rules-based model, try other neural network models and select other layers and add more neuron layers to further optimize the model.


## 10. Business Recommendation

It goes back to our original intention of creating this model – why we need to classify news. We know that different news will lead to different impact to the stock price movement. By combing news catalyst and the past stock movement using statistical approach, we will have an idea of how certain news type will drive the stock price in which direction, and by how much approximately. And the first step will always go back to correctly classifying the news type. The goal of this project is to create insights to retail and/or institutional investors and traders who make investment decisions based on event-driven strategies.
