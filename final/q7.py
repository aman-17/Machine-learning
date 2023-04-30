def eta_squared(counts, means, css):
    grand_mean = sum([count * mean for count, mean in zip(counts, means)]) / sum(counts)
    sst = sum(css) + sum([count * (mean - grand_mean) ** 2 for count, mean in zip(counts, means)])
    ssm = sum([count * (mean - grand_mean) ** 2 for count, mean in zip(counts, means)])
    eta_sq = ssm / sst
    return eta_sq

counts = [92, 226, 110]
means = [22.4673913043478, 29.5044247787611, 25.0363636363636]
css = [1574.9021739130400, 7794.4955752212400, 983.8545454545450]

eta_sq = eta_squared(counts, means, css)
print(f"Eta-squared: {eta_sq:.4f}")