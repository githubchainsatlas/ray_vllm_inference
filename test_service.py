import requests
import json

def test_request(url, request_data):
    """
    Test a request to the vLLM server
    
    Args:
        url: The URL to send the request to
        request_data: The data to send in the request
    """
    print(f"Testing request to {url}")
    print(f"Request data: {json.dumps(request_data)}")
    
    # Make the request
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=request_data
        )
        
        # Print the response
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response body: {response.text}")
        
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

# Define test cases
test_cases = [
    # Basic prompt test
    {"prompt": "The capital of France is ", "max_tokens": 32, "temperature": 0},
    
    # Message-based test
    {"messages": [{"role": "user", "content": "The capital of France is "}], "max_tokens": 32, "temperature": 0},
    
    # Test with content-type parameter explicitly
    {"prompt": "The capital of France is ", "max_tokens": 32, "temperature": 0, "content_type": "application/json"},
    
    # Test with different spacing
    {"prompt":"The capital of France is","max_tokens":32,"temperature":0}
]

# Run the tests
if __name__ == "__main__":
    url = "http://127.0.0.1:8000/generate"
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test case {i+1} ===")
        test_request(url, test_case)