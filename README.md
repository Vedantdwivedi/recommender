

**Product Retrieval API**

**Overview**

This is a product retrieval API that uses a TF-IDF retrieval model to retrieve products based on a search query. The API is built using FastAPI and uses a Pydantic model for the query and response.

**Getting Started**

To get started with this API, follow these steps:

1. **Clone the repository**: Clone this repository to your local machine using `git clone`.
2. **Install dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **Run the API**: Run the API using `uvicorn main:app --host 0.0.0.0 --port 8000`.

**API Endpoints**

The API has one endpoint:

* **`/search`**: This endpoint takes a search query as input and returns a list of top product IDs.

**Request Body**

The request body should be a JSON object with the following structure:

```json
{
  "query": "search query"
}
```

**Response**

The response will be a JSON object with the following structure:

```json
{
  "top_product_ids": ["product_id_1", "product_id_2", ...]
}
```

**Example Use Case**

Here's an example of how to use the API:

* **Search for products**: Send a POST request to `/search` with a search query, such as "armchair".
* **Get top product IDs**: The API will return a list of top product IDs that match the search query.

**Troubleshooting**

If you encounter any issues with the API, check the following:

* **Check the logs**: Check the logs to see if there are any errors or warnings.
* **Check the dependencies**: Make sure that all dependencies are installed correctly.
* **Check the API endpoint**: Make sure that the API endpoint is correct and that the request body is in the correct format.

**Contributing**

If you'd like to contribute to this project, please follow these steps:

* **Fork the repository**: Fork this repository to your own GitHub account.
* **Create a new branch**: Create a new branch for your changes.
* **Make changes**: Make changes to the code and commit them to your branch.
* **Create a pull request**: Create a pull request to merge your changes into the main branch.

**License**

This project is licensed under the MIT License. See the LICENSE file for more information.
