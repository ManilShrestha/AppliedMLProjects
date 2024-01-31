## Vision: Plant disease classification through photographs of leaves

Advancements in computer vision technology can be of great help to farmers and people interested in house (indoor) plants. With the use of this model, plant enthusiasts can identify any diseases that may need attention. The model created in this project can be incorporated into a mobile app that can provide the type of disease. With information on the type of disease, mobile developers can use a database to add necessary solutions, such as fertilizers or changes in growing conditions to the plant. Highly effective for farmers.

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. Data was downloaded from a Kaggle competition: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

To achieve high accuracy on my task, I utilized a ResNet-18 model that had been pre-trained on the large-scale ImageNet dataset. ResNet-18 has demonstrated outstanding performance on a variety of classification tasks, as shown by a 2018 ACM paper(https://dl.acm.org/doi/abs/10.1145/3194452.3194461) and other studies. [Khan et al., 2018]. Given its track record, I was confident that this pre-trained model would produce strong results on my own task.
![image](https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/adc06fac-ba97-43bf-b8c5-9d4c27fe9d79)

<img width="524" alt="image" src="https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/a1668c09-62f7-45b2-911b-8f0a5b7542ec">


## Recommender System: Anime Recommendation System
Any video streaming platform anime recommendation system would be eager to fund a recommender system since they want to provide personalized recommendations to their users, increase user engagement, improve user satisfaction and retention, or drive revenue through increased sales or subscriptions. Additionally, an anime recommendation system may be of interest to organizations or users who want to analyze user behavior and preferences to gain insights into their target audience.

This dataset was obtained from a Kaggle competition: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database. The recommendation data was compiled from www.myanimelist.net.

Since this was my first recommendation system modeling, I went with a simple but effective way which is through Singular Value Decomposition and used Surprise python library (Hug, 2020). Singular Value Decomposition (SVD) is a matrix factorization technique used in recommender systems to identify latent factors or features that capture the preferences and characteristics of users and items. This is a collaborative filtering approach, and I chose this method because it can effectively handle missing data and sparsity in user-item rating. Additionally, the team that won the Netflix grand prize in 2009 used SVD method along with ensemble of ML techniques. (Toscher et al., 2009).
<img width="854" alt="image" src="https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/b33f70db-6df1-4fed-ab96-b4a920511aba">


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
<img width="696" alt="image" src="https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/1cb26c85-01e3-4fdd-996c-b0faddbb722e">


## Gaming Project: Agent Plays Mountain Car with Deep Q-Learning
Organizations or users, particularly those in the gaming industry seeking innovative ways to
enhance player experiences, may be drawn to fund a Deep Q-learning based gaming project due
to its advanced AI capabilities and potential for heightened user engagement. The enhanced
dynamism and immersion offered by such an application can directly boost revenue generation,
making the investment an attractive one. Additionally, the prospect of long-term savings in
development costs, due to reduced need for manual coding of complex gaming scenarios offered
by a well-structured ML application, further strengthens the appeal for potential funding.

I used the gaming environment provided by OpenAI’s Gym library.
https://gymnasium.farama.org/environments/classic_control/mountain_car/
Note:
For this project, I utilized the approach detailed in PyTorch's Deep Q-Learning tutorial,
accessible at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
Reward mechanisms, neural network architecture, and hyperparameters were best modified to
suit the specifics of the 'Mountain Car' gaming environment.
I chose Deep Q-Learning as the model for this project because of its proven efficiency in dealing
with high-dimensional and complex environments, such as those found in many gaming
situations. The deep learning component allows the model to extract useful representations from
raw data, efficiently handling the intricacies of these problems. Furthermore, this model is highly
versatile, providing a balance between exploration and exploitation, which is crucial in
reinforcement learning scenarios.

![image](https://github.com/ManilShrestha/AppliedMLProjects/assets/20830075/f8706725-dc78-490e-b9b4-1e31ffc700c2)

