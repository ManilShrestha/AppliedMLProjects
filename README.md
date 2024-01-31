## Vision: Plant disease classification through photographs of leaves

Advancements in computer vision technology can be of great help to farmers and people interested in house (indoor) plants. With the use of this model, plant enthusiasts can identify any diseases that may need attention. The model created in this project can be incorporated into a mobile app that can provide the type of disease. With information on the type of disease, mobile developers can use a database to add necessary solutions, such as fertilizers or changes in growing conditions to the plant. Highly effective for farmers.

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. Data was downloaded from a Kaggle competition: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

To achieve high accuracy on my task, I utilized a ResNet-18 model that had been pre-trained on the large-scale ImageNet dataset. ResNet-18 has demonstrated outstanding performance on a variety of classification tasks, as shown by a 2018 ACM paper(https://dl.acm.org/doi/abs/10.1145/3194452.3194461) and other studies. [Khan et al., 2018]. Given its track record, I was confident that this pre-trained model would produce strong results on my own task.

## Recommender System: Anime Recommendation System
Any video streaming platform anime recommendation system would be eager to fund a recommender system since they want to provide personalized recommendations to their users, increase user engagement, improve user satisfaction and retention, or drive revenue through increased sales or subscriptions. Additionally, an anime recommendation system may be of interest to organizations or users who want to analyze user behavior and preferences to gain insights into their target audience.

This dataset was obtained from a Kaggle competition: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database. The recommendation data was compiled from www.myanimelist.net.

Since this was my first recommendation system modeling, I went with a simple but effective way which is through Singular Value Decomposition and used Surprise python library (Hug, 2020). Singular Value Decomposition (SVD) is a matrix factorization technique used in recommender systems to identify latent factors or features that capture the preferences and characteristics of users and items. This is a collaborative filtering approach, and I chose this method because it can effectively handle missing data and sparsity in user-item rating. Additionally, the team that won the Netflix grand prize in 2009 used SVD method along with ensemble of ML techniques. (Toscher et al., 2009).

## NLP Project: Reuters News Classification
Organizations such as news agencies, content management platforms, or digital marketing firms
could be interested in funding a multiclass news classification ML application, as it could
streamline their content organization, improve recommendation algorithms, and enhance user
experience by providing tailored content. Additionally, this ML application could potentially
increase efficiency by automating manual content categorization tasks, thus freeing up human
resources for other value-added activities.

The Reuters-21578 text categorization dataset is a widely recognized collection of documents
that initially appeared on the Reuters newswire in 1987. Compiled and indexed with categories
by personnel from Reuters Ltd. and the Carnegie Group, Inc., this dataset is easily accessible to
machine learning enthusiasts via the NLTK library.

The initial step involved preprocessing the data: tokenizing the news using nltk.word_tokenize,
extracting TF-IDF features from these tokens, and applying Singular Value Decomposition
(SVD) to reduce the feature set from 35658 to 5000, while preserving 95% of data variance.
Following this, a fully connected neural network comprising two hidden layers activated by a
ReLU function was trained using stochastic gradient descent and cross entropy loss. While the
model's performance was subpar with the initial 35k+ features, the application of dimensionality
reduction led to a light enhancement in results; additionally, I utilized a simpler sklearn model,
specifically the OneVsRestClassifier and Linear SVC, to further evaluate the model.
![image](https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/e7c5563d-adc0-43ea-8fdb-b8e49eafb9e6)


