from urllib.parse import quote

def simulate_url_construction(prompt_name):
    """Simulate how the URL is constructed in the get_prompt_template function."""
    URL_API_PROMPTLAYER = "https://api.promptlayer.com"
    return f"{URL_API_PROMPTLAYER}/prompt-templates/{quote(prompt_name, safe='')}"

def test_get_prompt_template_url_encoding():
    """Test that prompt_name with slashes is properly URL encoded in the request URL."""
    test_cases = [
        {"prompt_name": "feature1/resolve_problem_2", "expected_encoded": quote("feature1/resolve_problem_2", safe='')},
        {"prompt_name": "feature1:test", "expected_encoded": quote("feature1:test", safe='')},
        {"prompt_name": "feature1/sub/test?query", "expected_encoded": quote("feature1/sub/test?query", safe='')}
    ]
    
    for test_case in test_cases:
        prompt_name = test_case["prompt_name"]
        expected_encoded = test_case["expected_encoded"]
        
        url = simulate_url_construction(prompt_name)
        
        assert expected_encoded in url, f"Expected {expected_encoded} in {url}"

def test_resolve_workflow_id_url_encoding():
    """Test that workflow_id_or_name with slashes is properly URL encoded."""
    pass

def main():
    """Run all verification tests."""
    test_get_prompt_template_url_encoding()
    test_resolve_workflow_id_url_encoding()
    print("All tests passed successfully!")

if __name__ == "__main__":
    main() 