# Import the necessary libraries
import logging  # Library for logging
import pandas as pd  # Library for data manipulation and analysis
from sklearn.feature_extraction.text import TfidfVectorizer  # Library for text feature extraction
from sklearn.metrics.pairwise import cosine_similarity  # Library for calculating cosine similarity
from fastapi import FastAPI, HTTPException  # Library for building APIs
from pydantic import BaseModel  # Library for defining data models
import uvicorn  # Library for running the API

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Define a base class for retrieval models
class RetrievalModel:
    """
    Base class for retrieval models.

    Attributes:
    vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer
    tfidf_matrix (csr_matrix): TF-IDF matrix for the products
    """
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer  # Initialize the vectorizer
        self.tfidf_matrix = tfidf_matrix  # Initialize the TF-IDF matrix

    def get_top_products(self, query, top_n=10):
        """
        Get the top N products for a given query.

        Parameters:
        query (str): Search query
        top_n (int): Number of top products to return (default=10)

        Returns:
        top_product_indices (np.ndarray): Indices of the top N products
        """
        raise NotImplementedError  # This method should be implemented by subclasses

# Define a TF-IDF retrieval model
class TfidfRetrievalModel(RetrievalModel):
    """
    TF-IDF retrieval model.

    Attributes:
    vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer
    tfidf_matrix (csr_matrix): TF-IDF matrix for the products
    """
    def __init__(self, vectorizer, tfidf_matrix):
        super().__init__(vectorizer, tfidf_matrix)  # Call the base class constructor

    def get_top_products(self, query, top_n=10):
        """
        Get the top N products for a given query.

        Parameters:
        query (str): Search query
        top_n (int): Number of top products to return (default=10)

        Returns:
        top_product_indices (np.ndarray): Indices of the top N products
        """
        query_vector = self.vectorizer.transform([query])  # Transform the query into a TF-IDF vector
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()  # Calculate the cosine similarities
        top_product_indices = cosine_similarities.argsort()[-top_n:][::-1]  # Get the indices of the top N products
        return top_product_indices

# Define a data loader class
class DataLoader:
    """
    Data loader class.

    Attributes:
    query_file (str): Path to the query file
    product_file (str): Path to the product file
    label_file (str): Path to the label file
    """
    def __init__(self, query_file, product_file, label_file):
        self.query_file = query_file  # Initialize the query file path
        self.product_file = product_file  # Initialize the product file path
        self.label_file = label_file  # Initialize the label file path

    def load_data(self):
        """
        Load the data from the files.

        Returns:
        query_df (pd.DataFrame): Query data
        product_df (pd.DataFrame): Product data
        label_df (pd.DataFrame): Label data
        """
        query_df = pd.read_csv(self.query_file, sep='\t')  # Load the query data
        product_df = pd.read_csv(self.product_file, sep='\t')  # Load the product data
        label_df = pd.read_csv(self.label_file, sep='\t')  # Load the label data
        return query_df, product_df, label_df

# Define a service class for the retrieval API
class RetrievalService:
    """
    Service class for the retrieval API.

    Attributes:
    retrieval_model (RetrievalModel): Retrieval model instance
    data_loader (DataLoader): Data loader instance
    """
    def __init__(self, retrieval_model, data_loader):
        self.retrieval_model = retrieval_model  # Initialize the retrieval model
        self.data_loader = data_loader  # Initialize the data loader

    def get_top_products(self, query):
        """
        Get the top products for a given query.

        Parameters:
        query (str): Search query

        Returns:
        top_product_ids (list): IDs of the top products
        """
        try:
            query_df, product_df, label_df = self.data_loader.load_data()  # Load the data
            top_product_indices = self.retrieval_model.get_top_products(query)  # Get the top product indices
            top_product_ids = product_df.iloc[top_product_indices]['product_id'].tolist()  # Get the top product IDs
            return top_product_ids
        except Exception as e:
            logger.error(f"Error getting top products: {e}")  # Log the error
            raise HTTPException(status_code=500, detail="Internal Server Error")  # Raise an HTTP exception

# Define a Pydantic model for the query
class Query(BaseModel):
    """
    Pydantic model for the query.

    Attributes:
    query (str): Search query
    """
    query: str

# Define a Pydantic model for the response
class Response(BaseModel):
    """
    Pydantic model for the response.

    Attributes:
    top_product_ids (list): IDs of the top products
    """
    top_product_ids: list

# Create the FastAPI app
app = FastAPI()

# Create a data loader instance
data_loader = DataLoader("query.csv", "product.csv", "label.csv")

# Create a TF-IDF retrieval model instance
vectorizer = TfidfVectorizer()
product_df = pd.read_csv("product.csv", sep='\t')
product_df['combined_text'] = product_df['product_name'] + ' ' + product_df['product_description']
product_df['combined_text'] = product_df['combined_text'].fillna('')  # Replace NaN values with empty strings
tfidf_matrix = vectorizer.fit_transform(product_df['combined_text'])
retrieval_model = TfidfRetrievalModel(vectorizer, tfidf_matrix)

# Create a retrieval service instance
retrieval_service = RetrievalService(retrieval_model, data_loader)

# Define the API endpoint for searching products
@app.post("/search", response_model=Response)
def search(query: Query):
    """
    Search for products.

    Parameters:
    query (Query): Search query

    Returns:
    Response: Response containing the top product IDs
    """
    top_product_ids = retrieval_service.get_top_products(query.query)  # Get the top product IDs
    return Response(top_product_ids=top_product_ids)  # Return the response

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the API
