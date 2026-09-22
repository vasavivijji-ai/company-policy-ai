import requests


test_cases = [
    {
        "question": "How many vacation days do employees receive?",
        "expected_keywords": ["15", "vacation"]
    },
    {
        "question": "Do employees receive dental and vision insurance?",
        "expected_keywords": ["dental", "vision"]
    },
    {
        "question": "What is the maternity leave policy?",
        "expected_keywords": ["couldn't find"]
    }
]


API_URL = "http://127.0.0.1:8000/ask"


passed_count = 0
failed_count = 0


print("=" * 40)
print("AI EVALUATION")
print("=" * 40)


for test in test_cases:

    question = test["question"]
    expected_keywords = test["expected_keywords"]


    response = requests.get(
        API_URL,
        params={"question": question}
    )


    data = response.json()

    actual = data["answer"]

    actual_lower = actual.lower()


    passed = all(
        keyword.lower() in actual_lower
        for keyword in expected_keywords
    )


    print()
    print("Question:", question)
    print("Expected keywords:", expected_keywords)
    print("Actual:", actual)


    if passed:

        print("Result: PASS")
        passed_count += 1

    else:

        print("Result: FAIL")
        failed_count += 1


    print("-" * 40)


# Calculate total tests
total_tests = len(test_cases)


# Calculate accuracy
accuracy = (passed_count / total_tests) * 100


print()
print("=" * 40)
print("AI EVALUATION RESULTS")
print("=" * 40)

print("Total Tests:", total_tests)
print("Passed:", passed_count)
print("Failed:", failed_count)
print("Accuracy:", f"{accuracy:.0f}%")

print("=" * 40)