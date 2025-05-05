import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

class MovieRecommender:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.user_features = None
        self.movie_features = None
        
    def prepare_data(self, ratings_df, movies_df):
        """
        Prepare the data for the recommendation system.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame containing user ratings
            movies_df (pd.DataFrame): DataFrame containing movie information
        """
        # Encode user and movie IDs
        self.user_encoder.fit(ratings_df['userId'].unique())
        self.movie_encoder.fit(ratings_df['movieId'].unique())
        
        # Create user and movie features
        self.dataset = Dataset()
        self.dataset.fit(
            users=self.user_encoder.classes_,
            items=self.movie_encoder.classes_
        )
        
        # Create interactions matrix
        interactions, _ = self.dataset.build_interactions(
            [(self.user_encoder.transform([u])[0],
              self.movie_encoder.transform([i])[0],
              r) for u, i, r in zip(ratings_df['userId'],
                                  ratings_df['movieId'],
                                  ratings_df['rating'])]
        )
        
        # Create user features
        user_features = pd.get_dummies(ratings_df['userId']).values
        self.user_features = csr_matrix(user_features)
        
        # Create movie features
        movie_features = pd.get_dummies(movies_df['movieId']).values
        self.movie_features = csr_matrix(movie_features)
        
        return interactions
    
    def train(self, interactions, num_components=30, loss='warp', epochs=20):
        """
        Train the LightFM model.
        
        Args:
            interactions: Sparse matrix of user-item interactions
            num_components (int): Number of components in the model
            loss (str): Loss function to use ('warp', 'bpr', 'logistic')
            epochs (int): Number of epochs to train
        """
        self.model = LightFM(
            no_components=num_components,
            loss=loss,
            learning_schedule='adagrad'
        )
        
        self.model.fit(
            interactions,
            user_features=self.user_features,
            item_features=self.movie_features,
            epochs=epochs,
            num_threads=4
        )
    
    def recommend(self, user_id, n_recommendations=10):
        """
        Generate movie recommendations for a given user.
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie IDs
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get user index
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get scores for all movies
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=np.arange(len(self.movie_encoder.classes_)),
            user_features=self.user_features,
            item_features=self.movie_features
        )
        
        # Get top n recommendations
        top_items = np.argsort(-scores)[:n_recommendations]
        
        # Convert back to original movie IDs
        return self.movie_encoder.inverse_transform(top_items)
    
    def evaluate(self, interactions, k=10):
        """
        Evaluate the model using precision@k.
        
        Args:
            interactions: Sparse matrix of user-item interactions
            k (int): Number of top items to consider
            
        Returns:
            float: Precision@k score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        precision = 0
        n_users = interactions.shape[0]
        
        for user_id in range(n_users):
            # Get recommendations
            recommended_items = self.recommend(user_id, n_recommendations=k)
            
            # Get actual items the user has interacted with
            actual_items = interactions[user_id].nonzero()[1]
            
            # Calculate precision@k
            hits = len(set(recommended_items) & set(actual_items))
            precision += hits / k
            
        return precision / n_users 