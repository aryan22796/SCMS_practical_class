# Dataset: Counts for each fruit
data = {
    'Mango': {'Yellow': 350, 'Sweet': 450, 'Long': 0, 'Total': 650},
    'Banana': {'Yellow': 400, 'Sweet': 300, 'Long': 350, 'Total': 400},
    'Others': {'Yellow': 50, 'Sweet': 100, 'Long': 50, 'Total': 150}
}

# Total counts for all features
total_counts = {'Yellow': 800, 'Sweet': 850, 'Long': 400, 'Total': 1200}

# Step 1: Calculate priors P(Mango), P(Banana), P(Others)
priors = {
    'Mango': data['Mango']['Total'] / total_counts['Total'],
    'Banana': data['Banana']['Total'] / total_counts['Total'],
    'Others': data['Others']['Total'] / total_counts['Total']
}

# Step 2: Calculate likelihoods P(Feature | Class)
def calculate_likelihood(feature, fruit):
    return data[fruit][feature] / data[fruit]['Total']

# Step 3: Naive Bayes prediction
def predict(features):
    probabilities = {}

    # Calculate the probability for each fruit (Mango, Banana, Others)
    for fruit in data:
        # Start with the prior probability P(Class)
        prob = priors[fruit]

        # Multiply by the likelihoods for each feature
        for feature in features:
            prob *= calculate_likelihood(feature, fruit)

        # Store the result
        probabilities[fruit] = prob

    # Return the fruit with the highest probability
    return max(probabilities, key=probabilities.get)

# Example usage: Predict based on Yellow and Sweet (ignoring Long in this example)
features = ['Yellow', 'Sweet', 'Long']
predicted_fruit = predict(features)
print(f"Predicted fruit: {predicted_fruit}")
