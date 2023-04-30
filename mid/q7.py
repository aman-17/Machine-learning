def global_frequency_distance(freq_dist, cat1, cat2):
    """
    Calculate the Global Frequency Distance between two categories based on their frequency distribution.

    :param freq_dist: A dictionary representing the frequency distribution of a categorical feature.
                      The keys are the categories and the values are their respective frequencies.
    :param cat1: The first category.
    :param cat2: The second category.
    :return: The Global Frequency Distance between cat1 and cat2.
    """
    # Calculate the total number of observations
    N = sum(freq_dist.values())

    # Calculate the absolute difference between the frequencies of cat1 and cat2
    freq_diff = abs(freq_dist[cat1] - freq_dist[cat2])

    # Calculate the Global Frequency Distance between cat1 and cat2
    distance = freq_diff / N

    return distance

# Example usage
freq_dist = {
  'Others': 2388,
  'Professionals': 1276,
  'Office Worker': 948,
  'Management': 767,
  'Self-Employed': 193,
  'Retail Sales':109
}

cat1 = 'Management'
cat2 = 'Retail Sales'

distance = global_frequency_distance(freq_dist, cat1, cat2)
print(f'The Global Frequency Distance between {cat1} and {cat2} is {distance:.3f}')