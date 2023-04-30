from scipy.spatial.distance import cosine

focal_point = (3, 4)
points = [(-3, -4), (-1.5, 2), (-1, 0.75), (2, -1.5), (8, 6)]

min_distance = float('inf')
closest_point = None

for point in points:
    distance = cosine(focal_point, point)
    if distance < min_distance:
        min_distance = distance
        closest_point = point

print(f"Closest point: {closest_point}")